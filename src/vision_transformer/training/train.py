from dataclasses import asdict
from datetime import datetime

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from vision_transformer.config.experiment import ExperimentConfig
from vision_transformer.data.datasets import build_datasets
from vision_transformer.data.transforms import (
    build_eval_transforms,
    build_train_transforms,
)
from vision_transformer.logger.logger_factory import LoggerFactory
from vision_transformer.model.vision_transformer import VisionTransformer
from vision_transformer.training.losses import calculate_loss, get_criterion
from vision_transformer.training.lr_scheduler import get_lr_scheduler
from vision_transformer.training.optim import get_optimizer

LoggerFactory.configure()
_log = LoggerFactory.get_logger()


def training_loop(experiment_config: ExperimentConfig) -> None:

    _log.info(
        "> To see training related information such as loss, open"
        "tensorboard:  tensorboard --logdir=runs"
    )

    device = experiment_config.training.device
    epochs = experiment_config.training.epochs
    batch_size = experiment_config.training.batch_size
    gradient_accumulation_steps = experiment_config.training.gradient_accumulation_steps
    grad_clip_global_norm = experiment_config.training.grad_clip_global_norm
    dataloader_num_workers = experiment_config.training.dataloader_num_workers

    assert gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1"

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/{run_name}")

    # Get transforms for dataset
    transform_conf = experiment_config.transform
    train_trans = build_train_transforms(cfg=transform_conf)
    eval_trans = build_eval_transforms(cfg=transform_conf)

    # Get dataset and dataloaders
    dataset_conf = experiment_config.dataset
    train_ds, val_ds, test_ds = build_datasets(
        cfg=dataset_conf, train_transform=train_trans, eval_transform=eval_trans
    )
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
    )

    # Get loss functon
    loss_conf = experiment_config.loss
    criterion = get_criterion(**asdict(loss_conf))

    # Get vision transformer (ViT) model
    model_conf = experiment_config.model
    model = VisionTransformer(**asdict(model_conf), strict=True, head_type="pretrain")
    model.to(device=device)
    model.train()

    # Create optimizer
    optimizer_conf = experiment_config.optimizer
    optim = get_optimizer(params=model.parameters(), **asdict(optimizer_conf))

    # Learning rate scheduler
    lr_scheduler_conf = experiment_config.lr_scheduler
    steps_per_epoch = (
        len(train_dl) + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_steps = epochs * steps_per_epoch
    lr_scheduler = get_lr_scheduler(
        optimizer=optim, **asdict(lr_scheduler_conf), total_steps=total_steps
    )

    optimizer_step = 0  # real update step

    # Training loop
    for _epoch in range(epochs):

        model.train()
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Epoch {_epoch+1}/{epochs}", leave=True)

        for step, (X, y) in enumerate(pbar, start=1):

            X = X.to(device)
            y = y.to(device)

            logit_pred = model(X)

            loss = calculate_loss(criterion=criterion, logit_pred=logit_pred, y_gt=y)
            # Scale for accumulation so gradient magnitudes match large batch training
            (loss / gradient_accumulation_steps).backward()

            do_step = (step % gradient_accumulation_steps == 0) or (
                step == len(train_dl)
            )
            if do_step:
                if grad_clip_global_norm is not None:
                    clip_grad_norm_(model.parameters(), grad_clip_global_norm)

                optim.step()
                lr_scheduler.step()
                optim.zero_grad(set_to_none=True)

                # log train loss in tensorboard
                writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=loss.item(),
                    global_step=optimizer_step,
                )

                # log learning rate
                writer.add_scalar(
                    tag="Learning-rate",
                    scalar_value=optim.param_groups[0]["lr"],
                    global_step=optimizer_step,
                )

                optimizer_step += 1

    writer.close()

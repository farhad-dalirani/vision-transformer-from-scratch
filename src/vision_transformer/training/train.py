from collections import deque
from dataclasses import asdict

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

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

    device = experiment_config.training.device
    epochs = experiment_config.training.epochs
    batch_size = experiment_config.training.batch_size
    grad_clip_global_norm = experiment_config.training.grad_clip_global_norm
    dataloader_num_workers = experiment_config.training.dataloader_num_workers

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
    model.train()
    model.to(device=device)

    # Create optimizer
    optimizer_conf = experiment_config.optimizer
    optim = get_optimizer(params=model.parameters(), **asdict(optimizer_conf))

    # Learning rate scheduler
    lr_scheduler_conf = experiment_config.lr_scheduler
    total_steps = epochs * len(train_dl)
    lr_scheduler = get_lr_scheduler(
        optimizer=optim, **asdict(lr_scheduler_conf), total_steps=total_steps
    )

    global_step = 0
    n_period = 50
    window_losses = deque(maxlen=n_period)

    # Training loop
    for _epoch in range(epochs):

        model.train()

        epoch_loss_sum = 0.0
        epoch_steps = 0
        window_losses.clear()

        for step, (X, y) in enumerate(train_dl, start=1):

            X = X.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logit_pred = model(X)

            loss = calculate_loss(criterion=criterion, logit_pred=logit_pred, y_gt=y)
            loss.backward()
            if grad_clip_global_norm is not None:
                clip_grad_norm_(model.parameters(), grad_clip_global_norm)

            optim.step()
            lr_scheduler.step()

            # log train loss
            loss_train = float(loss.item())
            global_step += 1
            epoch_steps += 1
            epoch_loss_sum += loss_train
            window_losses.append(loss_train)
            if step % n_period == 0:
                avg_window = sum(window_losses) / len(window_losses)
                avg_epoch = epoch_loss_sum / epoch_steps
                lr = optim.param_groups[0]["lr"]

                _log.info(
                    "epoch=%d/%d step=%d/%d global_step=%d "
                    "loss_avg_epoch=%.6f loss_avg_last_%d=%.6f lr=%.6e",
                    _epoch + 1,
                    epochs,
                    step,
                    len(train_dl),
                    global_step,
                    avg_epoch,
                    n_period,
                    avg_window,
                    lr,
                )

        # Log at the end of epoch
        avg_epoch = epoch_loss_sum / max(epoch_steps, 1)
        avg_window = sum(window_losses) / max(len(window_losses), 1)
        lr = optim.param_groups[0]["lr"]
        _log.info(
            "epoch=%d/%d DONE loss_epoch=%.6f loss_last_%d=%.6f lr=%.6e",
            _epoch + 1,
            epochs,
            avg_epoch,
            n_period,
            avg_window,
            lr,
        )

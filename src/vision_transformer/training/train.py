from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import ImageDraw, ImageFont
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from vision_transformer.config.experiment import ExperimentConfig
from vision_transformer.config.utils import dataclass_from_dict
from vision_transformer.data.datasets import build_datasets
from vision_transformer.data.transforms import (
    build_eval_transforms,
    build_train_transforms,
)
from vision_transformer.inference.predict import (
    evaluate_classifier,
    inference_on_one_batch,
)
from vision_transformer.logger.logger_factory import LoggerFactory
from vision_transformer.model.vision_transformer import VisionTransformer
from vision_transformer.training.checkpoint import save_checkpoint
from vision_transformer.training.losses import calculate_loss, get_criterion
from vision_transformer.training.lr_scheduler import get_lr_scheduler
from vision_transformer.training.optim import get_optimizer
from vision_transformer.visualization.prediction import (
    log_model_predictions,
)

LoggerFactory.configure()
_log = LoggerFactory.get_logger()


def training_loop(experiment_config: ExperimentConfig) -> None:
    """Run the end-to-end training loop for the Vision Transformer.

    This function builds the full training pipeline from an ``ExperimentConfig``:
    transforms, datasets/dataloaders, model, loss, optimizer, and learning-rate
    scheduler. It supports resuming training from a checkpoint when
    ``experiment_config.training.resume_path`` is set.

    Resume behavior:
      - Loads the checkpoint dictionary early.
      - Reconstructs ``ExperimentConfig`` from the checkpoint and uses it as the
        source of truth for building the pipeline.
      - Restores model, optimizer, and scheduler states.
      - Moves optimizer state tensors (e.g., Adam moments) to the configured device.
      - Restores counters such as epoch index and optimizer step.

    Checkpointing behavior:
      - Saves ``last.pth`` at each evaluation point.
      - Saves ``best.pth`` when validation accuracy improves.

    Logging:
        - Writes TensorBoard scalars for training loss and learning rate
        at every optimizer step.
        - Logs validation metrics (accuracy, F1, precision, recall)
        at evaluation intervals.
        - Logs final test metrics after training completes.
        - Optionally logs prediction visualizations (sample images with
        ground-truth and predicted labels) if
        ``training.log_prediction_visualizations`` is enabled.
        - All logs are written under ``runs/<run_name>`` and can be
        visualized using TensorBoard.

    Args:
        experiment_config: Experiment configuration dataclass containing training,
            dataset, transform, model, optimizer, scheduler, and loss settings.

    Returns:
        None
    """
    _log.info(
        "> To see training related information such as loss, open "
        "tensorboard:  tensorboard --logdir=runs\n"
    )

    # ---------------------------------------------------------------------
    # 1) (Optional) Resume: load checkpoint early so the checkpoint config
    #    becomes the single source of truth for the run.
    # ---------------------------------------------------------------------
    resume_path = getattr(experiment_config.training, "resume_path", None)
    ckpt = None
    if resume_path:
        ckpt = torch.load(str(resume_path), map_location="cpu")
        experiment_config = dataclass_from_dict(
            ExperimentConfig, ckpt["experiment_config"]
        )

        _log.info("\n>  Loaded Experiment Configuration to Resume Training:\n")
        _log.info(experiment_config)
        _log.info("\n")

    # ---------------------------------------------------------------------
    # 2) Pull commonly-used training settings from the config
    # ---------------------------------------------------------------------
    device = experiment_config.training.device
    epochs = experiment_config.training.epochs
    batch_size = experiment_config.training.batch_size
    gradient_accumulation_steps = experiment_config.training.gradient_accumulation_steps
    grad_clip_global_norm = experiment_config.training.grad_clip_global_norm
    dataloader_num_workers = experiment_config.training.dataloader_num_workers
    eval_interval = experiment_config.training.eval_interval
    if experiment_config.training.checkpoints_dir is not None:
        checkpoints_dir = Path(experiment_config.training.checkpoints_dir)
    log_prediction_visualizations = (
        experiment_config.training.log_prediction_visualizations
    )
    normalization_mean = experiment_config.transform.mean
    normalization_std = experiment_config.transform.std
    num_classes = experiment_config.model.num_classes
    image_size = experiment_config.transform.image_size

    assert gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1"
    assert experiment_config.transform.image_size == experiment_config.model.image_size

    # ---------------------------------------------------------------------
    # 3) Run naming + TensorBoard logging
    #    - If resuming, keep the same run_name so logs/checkpoints stay together.
    # ---------------------------------------------------------------------
    run_name = (
        Path(resume_path).parent.name
        if resume_path
        else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    writer = SummaryWriter(f"runs/{run_name}")

    # ---------------------------------------------------------------------
    # 4) Data pipeline: transforms -> datasets -> dataloaders
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 5) Loss / model / optimizer / scheduler setup (all built from config)
    # ---------------------------------------------------------------------
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
    # When using gradient accumulation, the number of optimizer updates per epoch
    # is smaller than len(train_dl). We compute that here for scheduler total_steps.
    steps_per_epoch = (
        len(train_dl) + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_steps = epochs * steps_per_epoch
    lr_scheduler = get_lr_scheduler(
        optimizer=optim, **asdict(lr_scheduler_conf), total_steps=total_steps
    )

    # ---------------------------------------------------------------------
    # 6) Restore from checkpoint (if any)
    #    - Model weights.
    #    - Optimizer states often contain tensors (e.g., Adam moments) that must be
    #      moved to the same device as the model.
    # ---------------------------------------------------------------------
    start_epoch = 0
    optimizer_step = 0
    best_val_acc = float("-inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])

        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        if ckpt.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

        # Restore counters
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_step = ckpt.get("optimizer_step", 0)
        best_val_acc = ckpt.get("best_val_acc", float("-inf"))
        if best_val_acc is None:
            best_val_acc = float("-inf")

        _log.info(
            f"Resumed from {resume_path} at epoch={start_epoch}, optimizer_step={optimizer_step}"
        )

    # ---------------------------------------------------------------------
    # 7) Training loop
    #    - Uses gradient accumulation to simulate larger batch sizes.
    #    - Logs loss and LR on every optimizer update.
    # ---------------------------------------------------------------------
    for _epoch in range(start_epoch, epochs):

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

        # -----------------------------------------------------------------
        # 8) Validation + checkpointing
        #    - Save "best" when accuracy improves, always save "last".
        # -----------------------------------------------------------------
        if _epoch % eval_interval == 0 or _epoch == epochs - 1:
            val_set_metrics = evaluate_classifier(
                model=model,
                dataloader=val_dl,
                device=device,
                num_classes=num_classes,
                compute_confusion_matrix=False,
            )
            writer.add_scalar("Accuracy/val", val_set_metrics["accuracy"], _epoch)
            writer.add_scalar("F1/val", val_set_metrics["f1"], _epoch)
            writer.add_scalar("Precision/val", val_set_metrics["precision"], _epoch)
            writer.add_scalar("Recall/val", val_set_metrics["recall"], _epoch)

            val_acc = val_set_metrics["accuracy"]
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            # Save model
            if is_best:
                save_checkpoint(
                    checkpoints_dir / run_name / "best.pth",
                    epoch=_epoch,
                    optimizer_step=optimizer_step,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    experiment_config=experiment_config,
                    best_val_acc=best_val_acc,
                )
            save_checkpoint(
                checkpoints_dir / run_name / "last.pth",
                epoch=_epoch,
                optimizer_step=optimizer_step,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                experiment_config=experiment_config,
                best_val_acc=best_val_acc,
            )

    # ---------------------------------------------------------------------
    # 9) Final test evaluation
    # ---------------------------------------------------------------------
    test_set_metrics = evaluate_classifier(
        model=model,
        dataloader=test_dl,
        device=device,
        num_classes=num_classes,
        compute_confusion_matrix=False,
    )
    writer.add_scalar("Accuracy/test", test_set_metrics["accuracy"], _epoch)
    writer.add_scalar("F1/test", test_set_metrics["f1"], _epoch)
    writer.add_scalar("Precision/test", test_set_metrics["precision"], _epoch)
    writer.add_scalar("Recall/test", test_set_metrics["recall"], _epoch)

    # Optionally show output of model, for some test data
    if log_prediction_visualizations:
        log_model_predictions(
            writer=writer,
            model=model,
            test_ds=test_ds,
            device=device,
            step=step,
            batch_size=batch_size,
            image_size=image_size,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            inference_on_one_batch=inference_on_one_batch,
        )

    writer.close()

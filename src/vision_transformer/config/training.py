from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop.

    This dataclass defines hyperparameters and runtime options controlling
    optimization, evaluation frequency, checkpointing, and logging behavior
    during model training.

    Attributes:
        epochs (int):
            Total number of training epochs.

        batch_size (int):
            Number of samples per mini-batch.

        gradient_accumulation_steps (int):
            Number of forward/backward passes to accumulate gradients
            before performing an optimizer step. Useful for simulating
            larger effective batch sizes when GPU memory is limited.

        grad_clip_global_norm (float | None):
            Maximum global norm for gradient clipping. If None, gradient
            clipping is disabled.

        device (str):
            Device identifier used for training (e.g., "cuda:0", "cpu").

        dataloader_num_workers (int):
            Number of worker processes used by the DataLoader.

        eval_interval (int):
            Number of epochs between evaluation runs.

        checkpoints_dir (str):
            Directory where model checkpoints are saved.

        resume_path (str | None):
            Path to a checkpoint file to resume training from.
            If None, training starts from scratch.

        log_prediction_visualizations (bool):
            If True, logs sample prediction visualizations (e.g., images
            with predicted and ground-truth labels) to TensorBoard
            after finishing the training. Number of depicted images
            will be equal batch size.

        log_positional_embedding_visualizations (bool):
            If True, logs visualizations of the learned positional
            embeddings (norm maps and similarity heatmaps) to
            TensorBoard. Useful for inspecting spatial encoding behavior
            in Vision Transformers.

        log_attention_weights_visualizations (bool):
            If True, logs visualizations of the attention map for
            a batch of samples to Tensorboard after finishing the
            training. Number of depicted images will be equal batch
            size.
    """

    epochs: int = 1000
    batch_size: int = 128
    gradient_accumulation_steps: int = 16
    grad_clip_global_norm: float | None = 1.0
    device: str = "cuda:0"
    dataloader_num_workers: int = 2
    eval_interval: int = 20
    checkpoints_dir: str = "./checkpoints"
    resume_path: str | None = None

    log_prediction_visualizations: bool = True
    log_positional_embedding_visualizations: bool = True
    log_attention_map_visualizations: bool = True

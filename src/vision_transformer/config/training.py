from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    epochs: int = 1000
    batch_size: int = 128
    gradient_accumulation_steps: int = 16
    grad_clip_global_norm: float | None = 1.0
    device: str = "cuda:0"
    dataloader_num_workers: int = 2
    eval_interval: int = 20
    checkpoints_dir: str = "./checkpoints"
    resume_path: str | None = None

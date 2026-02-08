from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    epochs: int = 1000
    batch_size: int = 128
    gradient_accumulation_steps: int = 16
    grad_clip_global_norm: float | None = 1.0
    device = "cuda:0"
    dataloader_num_workers = 2
    eval_interval = 20
    checkpoints_dir = "./checkpoints"
    resume_path = './checkpoints/2026-02-07_21-15-26/last.pth'

from dataclasses import dataclass

from vision_transformer.config.dataset import DatasetConfig
from vision_transformer.config.loss import LossConfig
from vision_transformer.config.lr_schedular import LRSchedulerConfig
from vision_transformer.config.optim import OptimizerConfig
from vision_transformer.config.training import TrainingConfig
from vision_transformer.config.transform import TransformConfig
from vision_transformer.config.vit import ViTConfig


@dataclass(frozen=True)
class ExperimentConfig:
    model: ViTConfig
    transform: TransformConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    loss: LossConfig
    training: TrainingConfig

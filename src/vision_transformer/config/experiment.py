from dataclasses import dataclass

from vision_transformer.config.dataset import DatasetConfig
from vision_transformer.config.loss import LossConfig
from vision_transformer.config.lr_scheduler import LRSchedulerConfig
from vision_transformer.config.model import ViTConfig
from vision_transformer.config.optimizer import OptimizerConfig
from vision_transformer.config.training import TrainingConfig
from vision_transformer.config.transform import TransformConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration container for a training experiment.

    This dataclass aggregates all sub-configurations required to run
    a complete Vision Transformer experiment, including model,
    data processing, optimization, and training settings.

    Attributes:
        model (ViTConfig): Model architecture configuration.
        transform (TransformConfig): Data transformation and augmentation
            configuration.
        dataset (DatasetConfig): Dataset loading and dataset-specific
            parameters.
        optimizer (OptimizerConfig): Optimizer configuration.
        lr_scheduler (LRSchedulerConfig): Learning rate scheduler
            configuration.
        loss (LossConfig): Loss function configuration.
        training (TrainingConfig): Training loop and runtime settings
            (e.g., epochs, batch size, device).
    """

    model: ViTConfig
    transform: TransformConfig
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    loss: LossConfig
    training: TrainingConfig

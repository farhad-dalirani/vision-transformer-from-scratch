from dataclasses import dataclass
from typing import Literal

lossName = Literal["cross-entropy"]


@dataclass(frozen=True)
class LossConfig:
    """Configuration for constructing a training loss.

    This dataclass specifies the type of loss function and its
    associated configuration parameters.

    Attributes:
        loss_name (lossName): Name of the loss function to use.
            Currently supported values:
                - "cross-entropy": Standard cross-entropy loss.
            Defaults to "cross-entropy".
        label_smoothing (float): Amount of label smoothing to apply.
            Must be in the range [0.0, 1.0]. A value of 0.0 disables
            label smoothing. Defaults to 0.0.
        reduction (str): Specifies the reduction to apply to the output.
            Common options include:
                - "mean": Average the loss over the batch.
                - "sum": Sum the loss over the batch.
                - "none": No reduction.
            Defaults to "mean".
    """

    loss_name: lossName = "cross-entropy"
    label_smoothing: float = 0.0
    reduction: str = "mean"

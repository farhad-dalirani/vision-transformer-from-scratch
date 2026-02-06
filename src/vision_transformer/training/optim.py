from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer, ParamsT

from vision_transformer.config.optimizer import optimizerName


def get_optimizer(
    params: ParamsT, opt_name: optimizerName = "adam", **kwargs
) -> Optimizer:
    """
    Factory function to create a PyTorch optimizer.

    This function dispatches optimizer creation based on the provided
    optimizer name and hyperparameters.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining
            parameter groups.
        opt_name (optimizerName): Name of the optimizer to use. Supported
            values are `"adam"` and `"sgd"`.
        **kwargs: Additional optimizer-specific keyword arguments.

            For Adam:
                - lr (float, optional): Learning rate. Default is 8e-4.
                - betas (Tuple[float, float], optional): Coefficients used for
                  computing running averages of gradient and its square.
                  Default is (0.9, 0.999).
                - weight_decay (float, optional): Weight decay (L2 penalty).
                  Default is 0.0.

            For SGD:
                - lr (float, optional): Learning rate. Default is 8e-4.
                - momentum (float, optional): Momentum factor. Default is 0.9.
                - weight_decay (float, optional): Weight decay (L2 penalty).
                  Default is 0.0.

    Returns:
        Optimizer: Instantiated PyTorch optimizer.

    Raises:
        ValueError: If the requested optimizer name is not supported.
    """
    if opt_name.lower() == "adam":
        lr = kwargs.get("lr", 8e-4)
        betas = kwargs.get("betas", (0.9, 0.999))
        weight_decay = kwargs.get("weight_decay", 0.0)

        return get_adam_optimizer(
            params=params, lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif opt_name.lower() == "sgd":
        lr = kwargs.get("lr", 8e-4)
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.0)

        return get_sgd_optimizer(
            params=params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            f"Requested optimizer {opt_name}, is not supported,"
            f" should be {str(optimizerName)}"
        )


def get_adam_optimizer(
    params: ParamsT, lr: float, betas: tuple[float, float], weight_decay: float
) -> Optimizer:
    """
    Create an Adam optimizer.

    Args:
        params (ParamsT): Iterable of parameters to optimize.
        lr (float): Learning rate.
        betas (Tuple[float, float]): Coefficients for computing running averages
            of gradient and squared gradient.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        Optimizer: Adam optimizer instance.
    """
    return Adam(params=params, lr=lr, weight_decay=weight_decay, betas=betas)


def get_sgd_optimizer(
    params: ParamsT, lr: float, momentum: float, weight_decay: float
) -> Optimizer:
    """
    Create an SGD optimizer with momentum.

    Args:
        params (ParamsT): Iterable of parameters to optimize.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        Optimizer: SGD optimizer instance.
    """
    return SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)

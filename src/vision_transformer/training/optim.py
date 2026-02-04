from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer, ParamsT

from vision_transformer.config.optimizer import optimizerName


def get_optimizer(
    params: ParamsT, opt_name: optimizerName = "adam", **kwargs
) -> Optimizer:

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

    return Adam(params=params, lr=lr, weight_decay=weight_decay, betas=betas)


def get_sgd_optimizer(
    params: ParamsT, lr: float, momentum: float, weight_decay: float
) -> Optimizer:

    return SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)

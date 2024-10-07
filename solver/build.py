import logging

from utils.parse_module_str import parse_module_str

logger = logging.getLogger(__name__)


def build_optimizer(optimizer_cfg, model):
    """
    Build optimizer.

    Args:
        optimizer_cfg: solver configuration
        model: model
    Returns:
        optimizer: optimizer
    """
    modifiable_optimizer_cfg = optimizer_cfg.copy()
    # Create parameter groups and set learning rate for each group
    lr = modifiable_optimizer_cfg.pop("lr")
    lr_factor = modifiable_optimizer_cfg.pop("lr_factor")
    bias_lr_factor = modifiable_optimizer_cfg.pop("bias_lr_factor")
    weight_decay = modifiable_optimizer_cfg.pop("weight_decay")
    weight_decay_bias = modifiable_optimizer_cfg.pop("weight_decay_bias")

    param_groups = {
        "default": {
            "params": [],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "cross": {
            "params": [],
            "lr": lr * lr_factor,
            "weight_decay": weight_decay,
        },
        "bias": {
            "params": [],
            "lr": lr * bias_lr_factor,
            "weight_decay": weight_decay_bias,
        },
        "classifier": {
            "params": [],
            "lr": lr * lr_factor,
            "weight_decay": weight_decay,
        },
    }

    logger.info(f"Using {lr_factor} times learning rate for random init module ")

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        if "cross" in key:
            group = "cross"
        elif "bias" in key:
            group = "bias"
        elif "classifier" in key or "mlm_head" in key:
            group = "classifier"
        else:
            group = "default"

        param_groups[group]["params"].append(value)

    # Remove empty groups
    param_groups = [group for group in param_groups.values() if group["params"]]

    # Create optimizer
    optimizer_type = modifiable_optimizer_cfg.pop("type")
    logger.info(f"Using {optimizer_type} optimizer")
    optimizer_cls = parse_module_str(optimizer_type)
    optimizer = optimizer_cls(param_groups, **modifiable_optimizer_cfg)

    # if optimizer_cfg.optimizer == "SGD":
    #     optimizer = torch.optim.SGD(param_groups, momentum=optimizer_cfg.momentum)
    # elif optimizer_cfg.optimizer == "Adam":
    #     optimizer = torch.optim.Adam(
    #         param_groups,
    #         betas=(optimizer_cfg.alpha, optimizer_cfg.beta),
    #         eps=1e-3,
    #     )
    # elif optimizer_cfg.optimizer == "AdamW":
    #     optimizer = torch.optim.AdamW(
    #         param_groups,
    #         betas=(optimizer_cfg.alpha, optimizer_cfg.beta),
    #         eps=1e-8,
    #     )
    # elif optimizer_cfg.optimizer == "PagedAdamW32bit":
    #     optimizer = bitsandbytes.optim.PagedAdamW32bit(
    #         param_groups,
    #         betas=(optimizer_cfg.alpha, optimizer_cfg.beta),
    #         eps=1e-8,
    #     )
    # elif optimizer_cfg.optimizer == "PagedAdamW8bit":
    #     optimizer = bitsandbytes.optim.PagedAdamW8bit(
    #         param_groups,
    #         betas=(optimizer_cfg.alpha, optimizer_cfg.beta),
    #         eps=1e-8,
    #     )
    # elif optimizer_cfg.optimizer == "AdamW8bit":
    #     optimizer = bitsandbytes.optim.AdamW8bit(
    #         param_groups,
    #         betas=(optimizer_cfg.alpha, optimizer_cfg.beta),
    #         eps=1e-8,
    #     )
    # else:
    #     raise NotImplementedError

    return optimizer


def build_lr_scheduler(scheduler_cfg, optimizer):
    """Build learning rate scheduler.
    Args:
        scheduler_cfg: solver configuration
    Returns:
        scheduler: learning rate scheduler
    """
    modifiable_scheduler_cfg = scheduler_cfg.copy()
    scheduler_type = modifiable_scheduler_cfg.pop("type")
    logger.info(f"Using {scheduler_type} scheduler.")
    lr_scheduler = parse_module_str(scheduler_type)(
        optimizer, **modifiable_scheduler_cfg
    )
    return lr_scheduler


# def build_lr_scheduler(optimizer_cfg, optimizer):
#     return LRSchedulerWithWarmup(
#         optimizer,
# milestones=optimizer_cfg.milestones,
# gamma=solver_cfg.gamma,
# warmup_factor=solver_cfg.warmup_factor,
# warmup_epochs=solver_cfg.warmup_epochs,
# warmup_method=solver_cfg.warmup_method,
# total_epochs=solver_cfg.num_epoch,
# mode=solver_cfg.lrscheduler,
# target_lr=solver_cfg.target_lr,
# power=solver_cfg.power,
#     )

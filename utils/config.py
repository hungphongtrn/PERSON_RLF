from omegaconf import OmegaConf


def resolve_tuple(*args):
    return tuple(args)


def set_up_cfg():
    OmegaConf.register_new_resolver("tuple", resolve_tuple)

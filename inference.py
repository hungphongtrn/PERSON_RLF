from pathlib import Path

from omegaconf import OmegaConf
import torch

from utils.parse_module_str import parse_module_str

def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)
OmegaConf.register_new_resolver("eval", eval)

def load_backbone(ckpt_path: str | Path):
    """
    Load the backbone from the checkpoint file.

    Args:
        ckpt_path (str | Path): The path to the checkpoint file.
    Returns:
        backbone (dict): The backbone from the checkpoint file.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    backbone_statedict = {}
    for key, value in state_dict.items():
        if "model.backbone" in key:
            new_key = key.replace("model.backbone.", "")
            backbone_statedict[new_key] = value

    backbone_omega = ckpt["hyper_parameters"]["config"]["backbone"]
    backbone_cls = backbone_omega["type"]
    config_cls = backbone_omega["config_type"]
    model_args = {
        "text_config": backbone_omega["text_config"],
        "vision_config": backbone_omega["vision_config"],
    }
    config = parse_module_str(config_cls)(**model_args)
    model = parse_module_str(backbone_cls)(config=config)
    model.load_state_dict(backbone_statedict)
    return model

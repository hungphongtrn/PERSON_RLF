from pathlib import Path
import os

from safetensors.torch import save_file
import torch
import fire


def extract_backbone_from_ckpt(
    ckpt_path: str | Path, target_format: str = "safetensors", output_dir: str | Path = "."
) -> str | Path:
    """
    Extract the backbone from the checkpoint file.

    Args:
        ckpt_path (str | Path): The path to the checkpoint file.
        target_format (str): The target format to save the backbone. Default: "safetensors".
    Returns:
        target_path (str | Path): The path to the saved
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    backbone = {}
    for key, value in state_dict.items():
        if "model.backbone" in key:
            new_key = key.replace("model.backbone.", "")
            backbone[new_key] = value

    if output_dir is not None:
        target_dir = output_dir
    else:
        target_dir = os.path.dirname(ckpt_path)

    if target_format == "safetensors":
        target_path = os.path.join(target_dir, "backbone.safetensors")
        save_file(backbone, target_path)
    elif target_format == "pt":
        target_path = os.path.join(target_dir, "backbone.pt")
        torch.save(backbone, target_path)

    return target_path


if __name__ == "__main__":
    fire.Fire(extract_backbone_from_ckpt)

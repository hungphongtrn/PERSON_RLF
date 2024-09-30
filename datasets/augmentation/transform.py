"""
Image transformation functions.
"""

import logging
import random
from typing import Callable, Tuple
from functools import reduce

import numpy as np
from PIL import ImageFilter
from torchvision import transforms

from utils.parse_module_str import parse_module_str

logger = logging.getLogger(__name__)


def get_self_supervised_augmentation(img_size):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[0.1, 2.0]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), antialias=True),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return aug


def build_image_augmentation_pool(
    augment_cfg: dict = None,
    size: Tuple[int, int] = (384, 128),
    k: int = 2,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Build a pool of image augmentation functions.

    Args:
        augment_cfg (dict): A dictionary of image augmentation functions.
        size: The size of the output image.
        k: The number of augmentation functions to apply.
        is_train: Whether the augmentation is for training.
        If is not for training, only resize and normalize will be applied.
    Returns:
        A callable that applies a random selection of image augmentation functions.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image_aug = [transforms.Resize(size), transforms.ToTensor()]
    if is_train:
        if augment_cfg:
            for aug_type, aug_params in augment_cfg.items():
                aug = parse_module_str(aug_type)(**aug_params)
                image_aug.append(aug)

            aug_choice = np.random.choice(image_aug, k)
            # Extend image_aug with aug_choice
            image_aug.extend(aug_choice)

    # augment_cfg will not be used for testing
    image_aug.append(normalize)

    if is_train:
        logger.info(f"Using image augmentation: {image_aug} for training.")

    return transforms.Compose(image_aug)


def build_text_augmentation_pool(
    augment_cfg: dict = None, k: int = 1, is_train: bool = True
) -> Callable:
    """
    Build a pool of text augmentation functions.

    Args:
        augment_cfg: A dictionary of text augmentation functions.
        k: The number of augmentation functions to apply.
        is_train: Whether the augmentation is for training.
        If is not for training, no augmentation will be applied.
    Returns:
        A callable that applies a random selection of text augmentation functions.
        or None if no augmentation is specified
    """
    # If not training or no augmentation is specified, return identity function
    if not is_train or not augment_cfg:
        return None

    text_aug_choices = []

    for aug_type, aug_params in augment_cfg.items():
        aug = parse_module_str(aug_type)(**aug_params)
        text_aug_choices.append(aug)

    aug_choice = np.random.choice(text_aug_choices, k)
    logger.info(f"Using text augmentation: {aug_choice} for training.")

    # Return a function to apply the augmentation iteratively
    return lambda x: reduce(lambda x, f: f(x), aug_choice, x)
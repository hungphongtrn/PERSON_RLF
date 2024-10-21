import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from typing import List, Dict
import textwrap

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
REVERSE_MEAN = (-MEAN / STD).tolist()
REVERSE_STD = (1.0 / STD).tolist()


def visualize_test(wrong_predictions: List[Dict], tokenizer, k: int = 10):
    """
    Visualize k randomly selected predictions, showing each query, predicted images with PIDs, and correct image.

    Args:
        wrong_predictions: List of dictionaries with structure:
        {'query', 'img_1': {'image': tensor, 'pid': str}, ..., 'img_10': {...}, 'correct_img': {...}}
        tokenizer: Tokenizer for decoding the query
        k: Number of predictions to visualize
    """
    denormalize = T.Normalize(mean=REVERSE_MEAN, std=REVERSE_STD)
    toPIL = T.ToPILImage()
    transform = T.Compose([denormalize, toPIL])

    # Randomly select k predictions
    k_selected_for_visualization = np.random.choice(wrong_predictions, k, replace=False)
    total_columns = 12  # query + 10 predictions + correct_img

    # Create a figure with k rows (one for each query)
    fig = plt.figure(figsize=(24, 4 * k))
    fig.suptitle("Query-Image Predictions Visualization", fontsize=16, y=0.98)

    def wrap_text(text, width=30):
        """Wrap text to specified width"""
        lines = textwrap.wrap(text, width=width, break_long_words=True)
        return "\n".join(lines)

    for idx, pred_dict in enumerate(k_selected_for_visualization):
        base_pos = idx * total_columns + 1

        # 1. Display Query
        ax = plt.subplot(k, total_columns, base_pos)
        if isinstance(pred_dict["query"], torch.Tensor):
            query_text = tokenizer.decode(pred_dict["query"], skip_special_tokens=True)
        else:
            query_text = pred_dict["query"]

        ax.set_facecolor("#f0f0f0")
        wrapped_text = wrap_text(query_text)
        bbox_props = dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", lw=1)
        plt.text(
            0.5,
            0.5,
            f"Query:\n{wrapped_text}",
            horizontalalignment="center",
            verticalalignment="center",
            wrap=True,
            fontsize=10,
            bbox=bbox_props,
            transform=ax.transAxes,
        )
        ax.set_aspect("equal", adjustable="box")
        plt.axis("off")

        # 2. Display Predicted Images
        for img_idx in range(1, 11):  # Display img_1 through img_10
            ax = plt.subplot(k, total_columns, base_pos + img_idx)

            img_key = f"img_{img_idx}"
            img_data = pred_dict[img_key]
            img_tensor = img_data['image']
            pid = img_data['pid']

            if isinstance(img_tensor, torch.Tensor):
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor.squeeze(0)
                img = transform(img_tensor.cpu())
            else:
                img = img_tensor

            plt.imshow(img)
            plt.title(f"Pred {img_idx}\nPID: {pid}", fontsize=10, pad=5)
            plt.axis("off")

        # 3. Display Correct Image (at the end)
        ax = plt.subplot(k, total_columns, base_pos + 11)
        correct_img_data = pred_dict["correct_img"]
        correct_img_tensor = correct_img_data['image']
        correct_pid = correct_img_data['pid']

        if isinstance(correct_img_tensor, torch.Tensor):
            if len(correct_img_tensor.shape) == 4:
                correct_img_tensor = correct_img_tensor.squeeze(0)
            correct_img = transform(correct_img_tensor.cpu())
        else:
            correct_img = correct_img_tensor

        plt.imshow(correct_img)
        plt.title(f"Ground Truth\nPID: {correct_pid}", fontsize=10, pad=5)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def display_single_prediction(pred_dict: Dict, tokenizer):
    """
    Display a single prediction with its query, predicted images with PIDs, and correct image.

    Args:
        pred_dict: Dictionary containing:
            - query
            - img_1 through img_10: each a dict with 'image' and 'pid'
            - correct_img: dict with 'image' and 'pid'
        tokenizer: Tokenizer for decoding the query
    """
    denormalize = T.Normalize(mean=REVERSE_MEAN, std=REVERSE_STD)
    toPIL = T.ToPILImage()
    transform = T.Compose([denormalize, toPIL])

    total_columns = 12  # query + 10 predictions + correct_img

    # Create figure
    fig = plt.figure(figsize=(24, 4))

    def wrap_text(text, width=30):
        """Wrap text to specified width"""
        lines = textwrap.wrap(text, width=width, break_long_words=True)
        return "\n".join(lines)

    # 1. Display Query
    ax = plt.subplot(1, total_columns, 1)
    if isinstance(pred_dict["query"], torch.Tensor):
        query_text = tokenizer.decode(pred_dict["query"])
    else:
        query_text = pred_dict["query"]

    ax.set_facecolor("#f0f0f0")
    wrapped_text = wrap_text(query_text)
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", lw=1)
    plt.text(
        0.5,
        0.5,
        f"Query:\n{wrapped_text}",
        horizontalalignment="center",
        verticalalignment="center",
        wrap=True,
        fontsize=10,
        bbox=bbox_props,
        transform=ax.transAxes,
    )
    ax.set_aspect("equal", adjustable="box")
    plt.axis("off")

    # 2. Display Predicted Images
    for img_idx in range(1, 11):  # Display img_1 through img_10
        plt.subplot(1, total_columns, img_idx + 1)

        img_key = f"img_{img_idx}"
        img_data = pred_dict[img_key]
        img_tensor = img_data['image']
        pid = img_data['pid']

        if isinstance(img_tensor, torch.Tensor):
            if len(img_tensor.shape) == 4:
                img_tensor = img_tensor.squeeze(0)
            img = transform(img_tensor.cpu())
        else:
            img = img_tensor

        plt.imshow(img)
        plt.title(f"Pred {img_idx}\nPID: {pid}", fontsize=10, pad=5)
        plt.axis("off")

    # 3. Display Correct Image (at the end)
    ax = plt.subplot(1, total_columns, 12)
    correct_img_data = pred_dict["correct_img"]
    correct_img_tensor = correct_img_data['image']
    correct_pid = correct_img_data['pid']

    if isinstance(correct_img_tensor, torch.Tensor):
        if len(correct_img_tensor.shape) == 4:
            correct_img_tensor = correct_img_tensor.squeeze(0)
        correct_img = transform(correct_img_tensor.cpu())
    else:
        correct_img = correct_img_tensor

    plt.imshow(correct_img)
    plt.title(f"Ground Truth\nPID: {correct_pid}", fontsize=10, pad=5)
    plt.axis("off")

    plt.tight_layout()
    return fig
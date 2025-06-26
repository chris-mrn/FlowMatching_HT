import argparse
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to train generative models."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )
    return parser.parse_args()


def show_images(tensor, title=None, nrow=5, save_path=None):
    """
    Plot and optionally save a batch of images: tensor of shape (B, C, H, W)

    Args:
        tensor: torch.Tensor
        title: str, optional
        nrow: int, number of images per row in the grid
        save_path: str or Path, full path where to save the image (e.g. "results/sample.png")
    """
    # If tensor is on GPU, bring it to CPU
    tensor = tensor.detach().cpu()

    # Convert to grid of images
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True)

    # Convert to numpy
    np_img = grid.permute(1, 2, 0).numpy()

    # Create figure
    plt.figure(figsize=(nrow * 2, 2.5))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(np_img)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved image to: {save_path}")

    plt.close()  # Don't display inline if only saving

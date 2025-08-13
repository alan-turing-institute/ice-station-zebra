from datetime import date

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes, Figure
from PIL.ImageFile import ImageFile

from .convert import image_from_figure


def plot_sic_comparison(
    target: np.ndarray, prediction: np.ndarray, date: date
) -> list[ImageFile]:
    """Plot the comparison of target and prediction for sea ice concentration."""
    fig: Figure
    axs: list[Axes]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="compressed")
    # Ground truth
    z_range = np.linspace(0, 1, 100)
    ground_truth = axs[0].contourf(target, levels=z_range, cmap="viridis")
    axs[0].set_title("Ground truth")
    # Prediction
    axs[1].contourf(prediction, levels=z_range, cmap="viridis")
    axs[1].set_title("Prediction")
    # Colourbar
    plt.colorbar(
        ground_truth, ax=axs, orientation="vertical", ticks=np.linspace(0, 1, 11)
    )
    # Title
    date_string = date.strftime(r"%Y-%m-%d")
    fig.suptitle(f"Comparison at {date_string}")
    # Set properties on all axes
    for ax in axs:
        ax.axis("off")
        ax.set_aspect("equal")
    return [image_from_figure(fig)]

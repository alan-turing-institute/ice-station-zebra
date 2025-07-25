from datetime import date

import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageFile
from .convert import image_from_figure


def plot_sic_comparison(target: np.ndarray, prediction: np.ndarray, date: date) -> ImageFile:
    """Plot the comparison of target and prediction for sea ice concentration."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Ground truth
    z_range = np.linspace(0, 1, 100)
    ground_truth = axs[0].contourf(target, levels=z_range, cmap="viridis")
    axs[0].set_title("Ground truth")
    axs[0].axis("off")
    # Prediction
    axs[1].contourf(prediction, levels=z_range, cmap="viridis")
    axs[1].set_title("Prediction")
    # Colourbar
    fig.colorbar(ground_truth, ax=axs[1], orientation="vertical")
    # Title
    date_string = date.strftime(r"%Y-%m-%d")
    fig.suptitle(f"Comparison at {date_string}")
    return image_from_figure(fig)

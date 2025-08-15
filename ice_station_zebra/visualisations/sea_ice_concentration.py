from datetime import date
from pathlib import Path
import io
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes, Figure
import matplotlib.animation as animation
from PIL.ImageFile import ImageFile
import tempfile
import os

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

def video_sic_comparison(
        targets: np.ndarray,
        predictions: np.ndarray,
        dates: list[date],
        fps: int = 2, 
        format: str = "mp4"
        ) -> bytes:
    """
    Create a video comparing the target and prediction sequences 
    for sea ice concentration.

    Args:
        targets: The target sea ice concentration sequences.
        predictions: The prediction sea ice concentration sequences.
        dates: The dates of the sequences.
        fps: The frames per second of the video.
        format: The format of the video.
        output_type: The type of the output.
        output_path: The path to the output file.

    Returns:
        The video as bytes, as supported by wandb,or the path to the output file.
    """
    # Check that the target and prediction sequences have the same shape
    if targets.shape != predictions.shape:
        raise ValueError("The target and prediction sequences must have the same shape.")
    if len(targets) != len(predictions) or len(targets) != len(dates):
        raise ValueError("The target, prediction, and date sequences must have the same length.")
    
    n_timesteps = targets.shape[0]
    
    # Create figure and axes. Explicit positioning of axes to avoid overlap or moving in the animation. 
    fig = plt.figure(figsize=(12, 6))
    axs = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    fig.tight_layout() 

    # Set up a colour range consistent with static plots
    z_range = np.linspace(0, 1, 100)

    # Create initial plots to set up the colorbar ONCE
    initial_target = targets[0]
    initial_prediction = predictions[0]
    
    image1 = axs[0].contourf(initial_target, levels=z_range, cmap="viridis")
    image2 = axs[1].contourf(initial_prediction, levels=z_range, cmap="viridis")
    
    # Set titles (these don't change)
    axs[0].set_title("Ground Truth", fontsize=14)
    axs[1].set_title("Prediction", fontsize=14)
    
    # Create one shared colorbar
    cbar = plt.colorbar(image1, ax=axs, orientation="vertical", ticks=np.linspace(0, 1, 11), shrink=0.8)
        

    # Set properties on all axes (once)
    for ax in axs:
        ax.set_xlim(0, targets[0].shape[1])
        ax.set_ylim(0, targets[0].shape[0])
        ax.axis("off")
        ax.set_aspect("equal")

    def animate(frame_idx: int) -> tuple:
        """Animation function to update each frame."""

        # Clear the contour collections, not the entire axes
        for ax in axs:
            for collection in ax.collections:
                collection.remove()

        # Create contour plot for each frame
        target_frame = targets[frame_idx]
        prediction_frame = predictions[frame_idx]
        
        axs[0].contourf(target_frame, levels=z_range, cmap="viridis")
        axs[1].contourf(prediction_frame, levels=z_range, cmap="viridis")

        # Update main title with current date and time of frame
        date_string = dates[frame_idx].strftime(r"%Y-%m-%d")
        time_string = dates[frame_idx].strftime(r"%H:%M")
        fig.suptitle(f"Comparison at {date_string} {time_string}")

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_timesteps, interval=1000//fps, blit=False, repeat=True
    )

    # Use temporary file approach since FFMpegWriter needs a file path
    temp_file = None
    try:
        # Create temporary file with appropriate extension
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
            temp_path = temp_file.name

        # Choose writer based on format
        if format.lower() == "gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, codec='h264', bitrate=1800)
        
        # Save to temporary file
        anim.save(temp_path, writer=writer)
        
        # Read file back as bytes
        with open(temp_path, 'rb') as f:
            video_bytes = f.read()
            
        return video_bytes
        
    finally:
        # Clean up
        plt.close(fig)
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
            
    
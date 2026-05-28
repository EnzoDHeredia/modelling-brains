"""Heatmap utilities for visualization and data collection.

Provides a small collector class to accumulate positions and save/show a
2D histogram heatmap. This extracts the heatmap responsibilities out of
the main controller for clarity and testability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d


class HeatmapCollector:
    """Collects 2D positional data and generates heatmaps."""

    def __init__(self):
        self.x_data = []
        self.y_data = []

    def add(self, x, y):
        """Add a data point (x, y)."""
        self.x_data.append(x)
        self.y_data.append(y)

    def to_arrays(self):
        return np.array(self.x_data), np.array(self.y_data)

    def save(self, path, dpi=150, bins=20, range=[[-5, 5], [-5, 5]]):
        """Save heatmap to file. Prints a warning when empty."""
        if not self.x_data:
            print("Warning: no heatmap data to save")
            return

        x_arr, y_arr = self.to_arrays()
        save_heatmap(x_arr, y_arr, path, bins=bins, range=range, dpi=dpi)

    def show(self, block=True):
        """Display a quick heatmap window (uses hist2d)."""
        if not self.x_data:
            print("Warning: no heatmap data to show")
            return

        x_arr, y_arr = self.to_arrays()
        plt.figure(figsize=(8, 8))
        plt.hist2d(x_arr, y_arr, bins=20, range=[[-5, 5], [-5, 5]], cmap='viridis')
        plt.colorbar(label='Count')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Position Heatmap')
        plt.show(block=block)


def save_heatmap(pos_x, pos_y, path, bins=20, range=[[-5, 5], [-5, 5]], dpi=150):
    """Standalone function kept for compatibility with existing code.

    Saves a 2D histogram as an image file.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.hist2d(pos_x, pos_y, bins=bins, range=range, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position Heatmap')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

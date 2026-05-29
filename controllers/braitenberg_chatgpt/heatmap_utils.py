"""Utilidades para recolectar posiciones y generar mapas de calor."""

import os

import matplotlib.pyplot as plt
import numpy as np


class HeatmapCollector:
    """Guarda posiciones 2D y genera mapas de calor del recorrido."""

    def __init__(self):
        self.x_data = []
        self.y_data = []

    def add(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)

    def to_arrays(self):
        return np.array(self.x_data), np.array(self.y_data)

    def save(self, path, dpi=150, bins=20, range_limits=None):
        if not self.x_data:
            print("Warning: no hay datos para guardar el mapa de calor")
            return

        if range_limits is None:
            range_limits = [[-5, 5], [-5, 5]]

        x_arr, y_arr = self.to_arrays()
        save_heatmap(x_arr, y_arr, path, bins=bins, range_limits=range_limits, dpi=dpi)

    def show(self, block=True, bins=20, range_limits=None):
        if not self.x_data:
            print("Warning: no hay datos para mostrar el mapa de calor")
            return

        if range_limits is None:
            range_limits = [[-5, 5], [-5, 5]]

        x_arr, y_arr = self.to_arrays()
        plt.figure(figsize=(8, 8))
        plt.hist2d(x_arr, y_arr, bins=bins, range=range_limits, cmap="viridis")
        plt.colorbar(label="Cantidad")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Mapa de calor del recorrido")
        plt.show(block=block)


class HeatmapRecorder:
    """Coordina recoleccion, deteccion de atasco y salida del heatmap."""

    def __init__(self):
        self.collector = HeatmapCollector()

    @property
    def x_data(self):
        return self.collector.x_data

    @property
    def y_data(self):
        return self.collector.y_data

    def record_node_position(self, node):
        position = node.getPosition()
        self.collector.add(position[0], position[1])
        return position

    def check_stuck(self, step, window, min_distance):
        if step <= window or len(self.collector.x_data) <= window:
            return False, None

        dx = self.collector.x_data[-1] - self.collector.x_data[-window]
        dy = self.collector.y_data[-1] - self.collector.y_data[-window]
        displacement = np.sqrt(dx * dx + dy * dy)

        return displacement < min_distance, displacement

    def finish(self, path, open_on_windows=True, show=True):
        self.collector.save(path)
        print("Mapa de calor guardado en: {}".format(path))

        if open_on_windows:
            try:
                if os.name == "nt":
                    os.startfile(path)
            except Exception as error:
                print("No se pudo abrir automaticamente el mapa: {}".format(error))

        if show:
            self.collector.show()


def save_heatmap(pos_x, pos_y, path, bins=20, range_limits=None, dpi=150):
    """Guarda un histograma 2D como imagen."""
    if range_limits is None:
        range_limits = [[-5, 5], [-5, 5]]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.hist2d(pos_x, pos_y, bins=bins, range=range_limits, cmap="viridis")
    plt.colorbar(label="Cantidad")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Mapa de calor del recorrido")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

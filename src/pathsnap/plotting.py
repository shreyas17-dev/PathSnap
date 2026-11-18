"""Shared plotting style so every figure in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")  # headless: scripts save PNGs, never open windows
import matplotlib.pyplot as plt

SPLIT_COLORS = {"train": "#2a6fdb", "student_test": "#e8833a", "evaluator_test": "#7a7a7a"}


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    })
    return plt

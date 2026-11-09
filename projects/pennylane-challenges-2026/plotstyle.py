"""One shared look for every figure in this project."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# categorical slots, always used in this order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 7, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def label_end(ax, x, y, text, color, dx=4, dy=0):
    """Direct label at the end of a line, in ink, with a colored marker beside it."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", color=INK, fontsize=9.5)

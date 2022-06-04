import matplotlib.pyplot as plt
import numpy as np


def save_pdf(
    image: np.array,
    cmap: str,
    filename: str,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> None:
    """
    Create a matplotlib plot of `image` and save it to `filename`
    """
    plt.imshow(
        image,
        interpolation="none",
        cmap=cmap,
        extent=[min_x, max_x, min_y, max_y],
        aspect="auto",
    )
    plt.savefig(filename)

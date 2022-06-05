import numpy as np

from utils.base import BaseWorker

from .cython_worker import _cython_calculate  # pylint: disable=no-name-in-module


class Worker(BaseWorker):
    """
    Numba python implementation of Mandelbrot set calculation
    """

    def _calculate(self) -> np.array:
        return _cython_calculate(
            max_x=self.max_x,
            min_x=self.min_x,
            max_y=self.max_y,
            min_y=self.min_y,
            max_iterations=self.max_iterations,
            pixels_x=self.pixels_x,
            pixels_y=self.pixels_y,
        )

import numba
import numpy as np

from utils.base import BaseWorker


@numba.jit(nopython=True)
def _mandel(real: np.float64, imag: np.float64, max_iterations: np.int8) -> np.int8:
    """determines if a point is in the Mandelbrot set based on deciding if,
    after a maximum allowed number of iterations, the absolute value of
    the resulting number is greater or equal to 2."""
    z_real = 0.0
    z_imag = 0.0

    for i in range(0, max_iterations):
        z_real, z_imag = (
            z_real * z_real - z_imag * z_imag + real,
            2 * z_real * z_imag + imag,
        )
        if (z_real * z_real + z_imag * z_imag) >= 4:
            return i
    return -1


@numba.jit(parallel=True, nopython=True)
def _numba_calculate(
    pixels_x: np.int64,
    pixels_y: np.int64,
    max_x: np.float64,
    min_x: np.float64,
    min_y: np.float64,
    max_y: np.float64,
    max_iterations: np.int8,
):

    step_x = (max_x - min_x) / pixels_x
    step_y = (max_y - min_y) / pixels_y

    image = np.ones(shape=(pixels_y, pixels_x), dtype=np.int8)

    for x_i in numba.prange(0, pixels_x):
        for y_i in numba.prange(0, pixels_y):
            iterations = _mandel(
                real=min_x + (x_i + 0.5) * step_x,
                imag=min_y + (y_i + 0.5) * step_y,
                max_iterations=max_iterations,
            )
            image[y_i, x_i] = iterations
    return image


class Worker(BaseWorker):
    """
    Numba python implementation of Mandelbrot set calculation
    """

    def _calculate(self) -> np.array:
        return _numba_calculate(
            max_x=self.max_x,
            min_x=self.min_x,
            max_y=self.max_y,
            min_y=self.min_y,
            max_iterations=self.max_iterations,
            pixels_x=self.pixels_x,
            pixels_y=self.pixels_y,
        )

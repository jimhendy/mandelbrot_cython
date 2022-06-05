import numpy as np

from utils.base import BaseWorker


class Worker(BaseWorker):
    """
    Pure python implementation of Mandelbrot set calculation
    """

    @staticmethod
    def mandel(real: float, imag: float, max_iterations: int) -> int:
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

    def _calculate(self) -> np.array:

        step_x = (self.max_x - self.min_x) / self.pixels_x
        step_y = (self.max_y - self.min_y) / self.pixels_y

        image = np.ones(shape=(self.pixels_y, self.pixels_x), dtype=np.int8)

        x_loc = self.min_x + (step_x / 2)
        for x_i in range(self.pixels_x):

            y_loc = self.min_y + (step_y / 2)
            for y_i in range(self.pixels_y):
                iterations = self.mandel(
                    real=x_loc, imag=y_loc, max_iterations=self.max_iterations
                )
                image[y_i, x_i] = iterations
                y_loc += step_y

            x_loc += step_x
        return image

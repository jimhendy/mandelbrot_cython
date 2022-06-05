import cython
import numpy as np
from cython.parallel import prange


cdef int _mandel(double real, double imag, int max_iterations) nogil:
    """determines if a point is in the Mandelbrot set based on deciding if,
    after a maximum allowed number of iterations, the absolute value of
    the resulting number is greater or equal to 2."""
    cdef double z_real = 0.0
    cdef double z_imag = 0.0
    cdef int counter = 0

    for counter in range(0, max_iterations):
        z_real, z_imag = (
            z_real * z_real - z_imag * z_imag + real,
            2 * z_real * z_imag + imag,
        )
        if (z_real * z_real + z_imag * z_imag) >= 4:
            return counter
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _cython_calculate(
    int pixels_x,
    int pixels_y,
    double max_x,
    double min_x,
    double min_y,
    double max_y,
    int max_iterations,
):

    cdef double step_x = (max_x - min_x) / pixels_x
    cdef double step_y = (max_y - min_y) / pixels_y

    data = np.ones(shape=(pixels_y, pixels_x), dtype=np.intc)
    cdef int[:, ::1] data_view = data

    cdef int x_i = 0
    cdef int y_i = 0
    cdef int iterations

    for x_i in prange(0, pixels_x, nogil=True):
        for y_i in prange(0, pixels_y):
            iterations = _mandel(
                real=min_x + (x_i + 0.5) * step_x,
                imag=min_y + (y_i + 0.5) * step_y,
                max_iterations=max_iterations,
            )
            data_view[y_i, x_i] = iterations
    return data

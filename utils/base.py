import time
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from utils.plotting import save_pdf


def timer(func: Callable) -> Any:
    """
    Log the execution time of a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f'Function "{func.__name__}" took {end-start} s')
        return result

    return wrapper


class BaseWorker(ABC):
    """
    Base class for Mandelbrot calculation workers
    """

    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        pixels_x: int,
        pixels_y: int,
        max_iterations: int,
        cmap: str = "Reds",
        run_id: str = "unnamed",
    ):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.max_iterations = max_iterations
        self.cmap = cmap
        self.image = None
        self.run_id = run_id
        self.output_dir = Path(__file__).parents[1] / "output"

        self._setup()

    def _setup(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._log_config()

    @timer
    def save(self) -> None:
        """
        Save the `self.image` to a file
        """
        if self.image is None:
            raise RuntimeError("Image not yet calculated, cannot save")
        save_pdf(
            image=self.image,
            filename=self.output_dir / f"{self.run_id}.pdf",
            cmap=self.cmap,
            min_x=self.min_x,
            max_x=self.max_x,
            min_y=self.min_y,
            max_y=self.max_y,
        )

    @abstractmethod
    def _calculate(self):
        ...

    @timer
    def calculate(self, *args, **kwargs):
        """
        Calculate the Mandelbrot set image
        """
        logger.info("Starting calculation")
        self.image = self._calculate(*args, **kwargs)
        logger.info("Calcualtion complete")

    def _log_config(self):
        logger.info("Worker Config")
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            logger.info(f"{key}: {value}")

"""
Compare various methods of calculating the Mandelbrot Set.

Usage:
python main.py <method>
where <method> is a keu from the WORKERS dict. E.g. "pure"
"""

import argparse
import datetime
from typing import Dict

from loguru import logger

from numba_python import Worker as numba_python
from pure_python import Worker as pure_python
from utils import BaseWorker

RUN_ID = datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
WORKERS: Dict[str, type[BaseWorker]] = {"pure": pure_python, "numba": numba_python}

logger.add(f"logs/{RUN_ID}.log")
logger.info(f"New Run initiated: {RUN_ID}")

parser = argparse.ArgumentParser()
parser.add_argument(
    "method_name", choices=WORKERS.keys(), help="Compute worker to use."
)
parser.add_argument("--min_x", type=float, default=-2.1)
parser.add_argument("--max_x", type=float, default=0.7)
parser.add_argument("--min_y", type=float, default=-1.2)
parser.add_argument("--max_y", type=float, default=1.2)
parser.add_argument("--pixels_x", type=int, default=1_000)
parser.add_argument("--pixels_y", type=int, default=1_000)
parser.add_argument("--max_iterations", type=int, default=20)
parser.add_argument("--cmap", type=str, default="jet")
parser.add_argument(
    "--img_format", choices=["svg", "pdf", "png"], default="pdf", type=str
)


def main(
    worker_class: type[BaseWorker],
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    max_iterations: int,
    pixels_x: int,
    pixels_y: int,
    cmap: str,
    img_format: str,
):
    """
    Find points inside the Mandelbrot set and save an image
    """
    logger.info(f"Calculating using worker: {worker_class}")
    worker = worker_class(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        pixels_x=pixels_x,
        pixels_y=pixels_y,
        max_iterations=max_iterations,
        cmap=cmap,
        run_id=RUN_ID,
        img_format=img_format,
    )
    worker.calculate()
    worker.save()


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(f"Command line args: {args}")
    main(
        worker_class=WORKERS[args.method_name],
        min_x=args.min_x,
        min_y=args.min_y,
        max_x=args.max_x,
        max_y=args.max_y,
        pixels_x=args.pixels_x,
        pixels_y=args.pixels_y,
        max_iterations=args.max_iterations,
        cmap=args.cmap,
        img_format=args.img_format,
    )

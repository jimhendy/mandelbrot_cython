# pylint: disable=invalid-name
import datetime
import subprocess
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

from numba_python.worker import Worker

logger.add(f"logs/film_{datetime.datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')}.log")

CENTER = (-2, 0)
ZOOM = 0.01
N_OUTPUTS = 1000

min_x = -2.5
max_x = 1.2
min_y = -1.2
max_y = 1.2
max_iterations = 30
pixels_x = 10_000
pixels_y = 10_000

output_dir = Path(__file__).parent / "film"
output_dir.mkdir(exist_ok=True, parents=True)
tmp_dir = Path(__file__).parent / "output"

worker = Worker(
    min_x=min_x,
    min_y=min_y,
    max_x=max_x,
    max_y=max_y,
    pixels_x=pixels_x,
    pixels_y=pixels_y,
    max_iterations=max_iterations,
    cmap="jet",
    run_id="placeholder",
    img_format="png",
)


try:
    for run_id in tqdm(range(N_OUTPUTS)):

        worker.run_id = f"{run_id:04}"

        worker.calculate()
        worker.save()

        filename = f"{worker.run_id}.{worker.format}"
        output_loc = tmp_dir / filename
        dest = output_dir / filename
        (tmp_dir / filename).rename(dest)

        worker.min_x += ZOOM * (CENTER[0] - worker.min_x)
        worker.max_x -= ZOOM * (worker.max_x - CENTER[0])

        worker.min_y += ZOOM * (CENTER[1] - worker.min_y)
        worker.max_y -= ZOOM * (worker.max_y - CENTER[1])
except Exception as e:  # pylint: disable=broad-except
    logger.error(e)

ff_cmd = f"ffmpeg -y -framerate 10 -i {output_dir.name}/%04d.png -c:v copy out.mp4"
subprocess.call(ff_cmd, shell=True)

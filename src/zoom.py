# pylint: disable=invalid-name
import datetime
import subprocess
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

from workers.cython_python.worker import Worker

logger.add(f"logs/film_{datetime.datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')}.log")

CENTER = (-0.725, -0.21)
ZOOM = 0.03
N_OUTPUTS = 10_000

min_x = -2.5
max_x = 1.2
min_y = -1.2
max_y = 1.2
max_iterations = 500
pixels_x = 10_000
pixels_y = 10_000

output_dir = Path(__file__).parents[1] / "film"
output_dir.mkdir(exist_ok=True, parents=True)
tmp_dir = Path(__file__).parents[1] / "output"

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

        dest = output_dir / worker.filename.name
        if (dest).exists():
            logger.info(f"Output ({dest.name})exists, skipping")
        else:
            worker.calculate()
            worker.save()

            output_loc = tmp_dir / worker.filename
            output_loc.rename(dest)

        worker.min_x += ZOOM * (CENTER[0] - worker.min_x)
        worker.max_x -= ZOOM * (worker.max_x - CENTER[0])

        worker.min_y += ZOOM * (CENTER[1] - worker.min_y)
        worker.max_y -= ZOOM * (worker.max_y - CENTER[1])
except Exception as e:  # pylint: disable=broad-except
    logger.error(e)

ff_cmd = f"ffmpeg -y -framerate 24 -i {output_dir.name}/%04d.png out.mp4"
subprocess.call(ff_cmd, shell=True)

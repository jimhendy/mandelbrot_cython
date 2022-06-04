"""
Docstring ToDo
"""

import argparse

import pure_python

WORKERS = {"pure": pure_python}

parser = argparse.ArgumentParser()
parser.add_argument("method", choices=WORKERS.keys(), help="Compute worker to use.")


def main(worker):
    """
    ToDo
    """
    print(worker)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        worker=WORKERS[args.method],
    )

from __future__ import annotations

import argparse
from typing import Sequence

from . import environment, infer, train, yolo_infer, yolo_train


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magicnet",
        description="MagicNet 2D command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a detector.")
    train.configure_parser(train_parser)
    train_parser.set_defaults(handler=train.run_from_args)

    infer_parser = subparsers.add_parser("infer", help="Run inference.")
    infer.configure_parser(infer_parser)
    infer_parser.set_defaults(handler=infer.run_from_args)

    env_parser = subparsers.add_parser("check-env", help="Inspect the environment and print install steps.")
    environment.configure_parser(env_parser)
    env_parser.set_defaults(handler=environment.run_from_args)

    yolo_train_parser = subparsers.add_parser("yolo-train", help="Train an Ultralytics YOLO detector.")
    yolo_train.configure_parser(yolo_train_parser)
    yolo_train_parser.set_defaults(handler=yolo_train.run_from_args)

    yolo_infer_parser = subparsers.add_parser("yolo-infer", help="Run Ultralytics YOLO inference.")
    yolo_infer.configure_parser(yolo_infer_parser)
    yolo_infer_parser.set_defaults(handler=yolo_infer.run_from_args)

    args = parser.parse_args(argv)
    return args.handler(args)

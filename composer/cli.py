from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Optional

from .pipeline import process_image

__all__ = ["build_arg_parser", "main"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relic Composer")
    parser.add_argument("--input", default="input", help="Input folder")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--use_midas", action="store_true", help="Run MiDaS depth")
    parser.add_argument("--alpha_model", default="u2net", help="Alpha model to use")
    parser.add_argument("--use_iiwnet", action="store_true", help="Run intrinsic decomposition")
    parser.add_argument("--triptych", action="store_true", help="Export triptych view")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg")
    files: Iterable[Path] = itertools.chain.from_iterable(
        in_dir.glob(pat) for pat in patterns
    )
    for img_file in files:
        process_image(img_file, out_dir, args)

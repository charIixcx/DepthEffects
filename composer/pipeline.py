from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2

from .io import load_image, save_image
from .ml import generate_depth, depth_to_normal, generate_alpha, decompose_albedo_shading

__all__ = ["process_image"]


def _create_depth(image: np.ndarray, use_midas: bool) -> np.ndarray:
    return generate_depth(image) if use_midas else np.zeros(image.shape[:2], dtype=np.uint8)


def process_image(path: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """Process a single image path."""
    image = load_image(path)
    depth_path = out_dir / "passes" / f"{path.stem}_depth.png"
    normal_path = out_dir / "passes" / f"{path.stem}_normal.png"
    alpha_path = out_dir / "passes" / f"{path.stem}_alpha.png"
    albedo_path = out_dir / "passes" / f"{path.stem}_albedo.png"
    shading_path = out_dir / "passes" / f"{path.stem}_shading.png"

    if depth_path.exists():
        depth = load_image(depth_path)
    else:
        depth = _create_depth(image, args.use_midas)
        save_image(depth, depth_path)

    if not normal_path.exists():
        normal = depth_to_normal(depth)
        save_image(normal, normal_path)
    if not alpha_path.exists():
        alpha = generate_alpha(image, model=args.alpha_model)
        save_image(alpha, alpha_path)
    if args.use_iiwnet and (not albedo_path.exists() or not shading_path.exists()):
        albedo, shading = decompose_albedo_shading(image)
        if not albedo_path.exists():
            save_image(albedo, albedo_path)
        if not shading_path.exists():
            save_image(shading, shading_path)

    normal = load_image(normal_path)
    edges = cv2.Canny(normal, 100, 200)
    composite = image.copy()
    composite[edges > 0] = [255, 0, 0]
    save_image(composite, out_dir / f"{path.stem}_composite.png")

"""Command-line tool for generating depth and compositing passes.

This module implements a minimal skeleton pipeline inspired by the README.
It loads portrait images from an input folder, generates or loads passes,
and composites them. Heavy ML operations are stubbed out but structured
so they can be replaced with real implementations later.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional
import itertools

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
except ImportError:  # pragma: no cover - modules are optional
    torch = None  # type: ignore


# ------------------------------- Utility functions -------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image into a NumPy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def generate_depth(image: np.ndarray) -> np.ndarray:
    """Generate a depth map using MiDaS if available; otherwise return zeros."""
    if torch is None:
        return np.zeros(image.shape[:2], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(
        "intel-isl/MiDaS",
        "DPT_Hybrid",
        pretrained=True,
    )
    model.eval().to(device)
    transform = Compose([
        Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal"),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    img_input = transform({"image": image})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        depth = prediction.squeeze().cpu().numpy()
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype(np.uint8)


def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    """Approximate normals from a depth map."""
    depth_f = depth.astype(np.float32) / 255.0
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack((-gx, -gy, np.ones_like(depth_f)))
    n = cv2.normalize(normal, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return n.astype(np.uint8)


def generate_alpha(image: np.ndarray, model: str = "u2net") -> np.ndarray:
    """Return a dummy alpha mask (placeholder for U2Net/SAM)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return mask


def decompose_albedo_shading(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate albedo and shading using a simple intrinsic decomposition."""
    img_f = image.astype(np.float32) / 255.0
    shading = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    shading = cv2.GaussianBlur(shading, (0, 0), sigmaX=5, sigmaY=5)
    shading = np.clip(shading, 0.1, 1.0)
    shading_rgb = np.repeat(shading[..., None], 3, axis=2)
    albedo = np.clip(img_f / shading_rgb, 0, 1)
    return (albedo * 255).astype(np.uint8), (shading * 255).astype(np.uint8)


# ------------------------------- Core Pipeline -------------------------------

def _create_depth(image: np.ndarray, use_midas: bool) -> np.ndarray:
    """Return a depth map using MiDaS if requested, else zeros."""
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

    # Basic composite: overlay normal map edges for demonstration
    normal = load_image(normal_path)
    edges = cv2.Canny(normal, 100, 200)
    composite = image.copy()
    composite[edges > 0] = [255, 0, 0]
    save_image(composite, out_dir / f"{path.stem}_composite.png")


# ------------------------------- CLI -------------------------------

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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

try:
    import torch
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
except ImportError:  # pragma: no cover - optional
    torch = None  # type: ignore

__all__ = ["generate_depth", "depth_to_normal", "generate_alpha", "decompose_albedo_shading"]

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

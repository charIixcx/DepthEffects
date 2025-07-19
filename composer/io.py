from pathlib import Path
from PIL import Image
import numpy as np

__all__ = ["load_image", "save_image"]

def load_image(path: Path) -> np.ndarray:
    """Load an image into a NumPy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save a NumPy array as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))

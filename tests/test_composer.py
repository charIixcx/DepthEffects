import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import numpy as np
import cv2
import relic_composer as rc


def test_generate_alpha_binary(tmp_path):
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    mask = rc.generate_alpha(img)
    unique = np.unique(mask)
    assert set(unique).issubset({0, 255})

def test_process_image_without_midas_skips_depth(monkeypatch, tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "input.png"
    rc.save_image(img, img_path)

    called = False

    def fake_generate_depth(image):
        nonlocal called
        called = True
        return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(rc, "generate_depth", fake_generate_depth)

    args = argparse.Namespace(use_midas=False, alpha_model="u2net", use_iiwnet=False, triptych=False)

    rc.process_image(img_path, tmp_path, args)

    assert not called
    assert (tmp_path / "passes" / "input_depth.png").exists()

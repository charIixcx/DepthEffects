📜 Epic of the Outlaw Saint Alchemist: Fully Automated Relic ComposerFrom one lone portrait to a mythic triptych of layered prophecies—no manual prep needed. Each chapter is a ritual task, each pass a shard of visual scripture. Invoke AI oracles, forge your pipeline, then summon divine relics.

# 🕯️ Chapter I: Single-Source Invocation

Place your portrait image (.png/.jpg) into the `/input/` folder.
Sample `input/` and `output/` directories with README files are provided for
testing the pipeline.

The system will automatically detect missing passes and use AI models to generate them.

---

## ✨ Chapter II: Gathering of Relics (Auto-Generated Passes)

- **Depth Map**: Generated via MiDaS (DPT_Large or Hybrid).
- **Normal Map**: Computed from depth gradients.
- **Alpha Mask**: Created using U²-Net or Segment Anything (SAM).
- **Albedo & Shading**: Derived via IIWNet or intrinsic decomposition.
- **Specular & Roughness**: Estimated with edge-detection and DNN filters.
- **Metallic Map**: Derived from highlight intensity or a learned network.

---

## ⚙️ Chapter III: Invocation Engine Setup

1. Install the pinned dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
   The `u2net` package is optional and may require manual installation via
   `pip install git+https://github.com/NathanUA/U-2-Net.git` if not available on
   PyPI.
2. Scaffold `relic_composer.py` with CLI flags:
    - `--input` (input folder)
    - `--output` (output folder)
    - `--use_midas`
    - `--alpha_model` (u2net / sam)
    - `--triptych`

---

## 💫 Chapter IV: Ritual of Light & Shadow (Compositing)

1. Load all passes into NumPy/OpenCV arrays.
2. **Rim Lighting**:
    - Compute `dot(normal, light_dir)` → glow mask.
    - Blend with basecolor using Soft Light mode.
3. **Gleam & Grain**:
    - Merge specular + metallic for relic shine.
    - Overlay grain via roughness map.
4. **Atmospheric Haze**:
    - Blur background based on depth.
    - Tint haze using shading map.

---

## 🩸 Chapter V: Transformation Masking

- Apply AI-generated alpha to isolate the figure.
- Add edge-ghosting for apparition effects.
- Save intermediate passes for remixing.

---

## 📁 Chapter VI: Archive of Visions (Exports)

- **Composite Output**: `/output/composite.png`
- **Passes Directory**: `/output/passes/`
  - `depth.png`
  - `normal.png`
  - `albedo.png`
  - `specular.png`
  - etc.
- **Triptych Output** (if `--triptych` is enabled):
  - Basecolor with grain.
  - Final composite.
  - Embossed normal + specular panel.

---

## 🔧 Chapter VII: Modular Configuration

Use `config.yaml` or CLI flags to customize:
- Depth oracle (render vs. MiDaS).
- Alpha model (U²-Net vs. SAM).
- Rim-light angle and intensity.
- Haze density and tint.
- Grain amount.

---

## 🕸 Chapter VIII: Apothecary of PyTorch (Optional Augmentations)

- Super-Resolution via ESRGAN.
- Denoising with DnCNN or RIDNet.
- Style Transfer using AdaIN or SPADE.
- Inpainting with LaMa or DeepFill-v2.
- Pose Detection via OpenPose or HRNet.
- Diffusion Fusion with Stable Diffusion + ControlNet.

---

## 🚀 Next Steps & Task Breakdown

- [x] Draft `relic_composer.py` skeleton with CLI parser.
- [x] Integrate MiDaS depth inference module.
- [x] Script depth → normal conversion.
- [x] Plug in U²-Net or SAM for auto-alpha.
- [x] Add IIWNet for albedo-shading split.
- [ ] Implement specular/roughness estimation.
- [ ] Composite passes (rims, gleam, haze) with OpenCV.
- [ ] Export outputs and triptychs.
- [x] Document usage in `README.md`.
- [x] Provide sample `/input/` and `/output/` for testing.
- [ ] *(Bonus)* Build a Streamlit GUI for live preview.

Forge your code in this order, and any input portrait will ascend into a layered relic—fully automated, endlessly remixable. What artifact shall we summon first?


## Usage

Install dependencies and run the composer on images placed in `input/`.
The processing pipeline lives in the `composer/` package with a thin wrapper
`relic_composer.py` for CLI usage:

```bash
pip install -r requirements.txt
python relic_composer.py --use_midas --alpha_model u2net --use_iiwnet
```

Results are stored in `output/` with generated passes under `output/passes/`.

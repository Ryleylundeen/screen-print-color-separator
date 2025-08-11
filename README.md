# Screen Print Color Separator

A minimal web app to convert a logo/image into screen print-ready color separations. Upload an image and the app will:

- Auto-detect the number of colors using an elbow method in a LAB-like color space
- Generate per-color plates with cleaned/dilated masks for films
- Generate an optional underbase with adjustable spread
- Provide colorized previews and original-color layers
- Package everything into a ZIP with a manifest and a composite that matches the source

## Features

- Automatic color count (up to 8)
- Perceptual clustering (LAB-like transform, k-means)
- Cleaned print plates: min-area removal and optional dilation
- Underbase generation with adjustable spread
- Original-color layers to reconstruct the exact image
- Colorized layers for quick review
- Single-click ZIP download with organized folders

## How it works

1. The image is converted to RGBA and sampled in a LAB-like color space
2. The optimal number of colors is estimated via an elbow method on SSE vs K
3. K-means clusters the sampled points, then the full image is labeled by nearest cluster
4. Two sets of masks are produced:
   - Raw masks: used to slice original per-pixel color layers (perfect visual composite)
   - Production masks: optional cleanup (min-area), and dilation for registration/bleed
5. Exports:
   - layers_original/: original per-pixel RGBA per cluster (stacks to the exact original)
   - layers_color/: flat color per cluster for visual reference
   - plates_black/: opaque black plates for film positives
   - underbase/: white and black variants (optional)
   - composite.png: layers_original composited server-side (reference)

## Output ZIP structure

- layers_original/
- layers_color/
- plates_black/
- underbase/ (optional)
- composite.png
- README.txt

## Requirements

- Python 3.10+

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
export PORT=5050
python app.py
# Open http://localhost:5050
```

## Usage

- Upload PNG/JPG/WebP
- Options:
  - Plate expand px: dilates production masks for registration tolerance
  - Min area (px): removes tiny specks from production masks
  - Underbase: adds a solid underbase with configurable spread
- Press Process and download the ZIP

### Photoshop recomposite

- Import all images from `layers_original/` as layers
- Blend mode Normal, opacity 100%, order as-is
- Ensure sRGB workflow without profile conversion
- Result should match `composite.png` and the original

## Notes for screen printing

- Use `plates_black/` for film positives
- `layers_color/` previews assist visual checks and spot-color naming
- `underbase/` gives both white and black variants; adjust spread based on garment and ink
- If your artwork includes gradients/anti-aliasing, keep `layers_original/` for reference and use halftones when burning screens

## Limitations

- Clustering is unsupervised; complex photos may separate into more plates than ideal
- The LAB-like transform is an approximation to avoid heavy native dependencies
- No RIP/halftone generation; use your shop’s RIP settings

## Security

- Uploads are processed in-memory and written under `runs/<uuid>/`
- `runs/` is ignored by git and not intended for permanent storage

## License

MIT

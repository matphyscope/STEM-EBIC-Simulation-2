"""Utilities for modeling a STEM image.

Workflow
--------
1. Load a .tif image.
2. Locate the scale bar (opaque black region in the bottom-left) and measure
   its pixel length so that a physical scale (nm/pixel) can be assigned.
3. Build the modeling region: everything that is opaque and does NOT belong
   to the scale-bar area.
4. Segment the modeling region by color (each distinct RGB colour becomes a
   candidate region).
5. Interactively let the user pick regions and name them (via ipywidgets in
   a Jupyter notebook).
6. Save an annotated image that shows the named regions plus a scale bar.

The module is designed to be driven step-by-step from a notebook so the user
controls each decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScaleBar:
    """Geometry and physical length of the scale bar."""

    bbox: Tuple[int, int, int, int]          # (y0, y0_end, x0, x0_end) of the ink region
    bar_row: int                              # image row where the horizontal bar sits
    bar_x0: int                               # left end of the bar in pixels
    bar_x1: int                               # right end of the bar in pixels
    pixel_length: int                         # bar length in pixels
    physical_length: Optional[float] = None   # user supplied, e.g. 100
    unit: str = "nm"

    @property
    def scale(self) -> float:
        """Physical units per pixel."""
        if self.physical_length is None:
            raise ValueError("physical_length not set; call set_physical_length first.")
        return self.physical_length / self.pixel_length


@dataclass
class Region:
    """A connected colour region in the image."""

    index: int
    color: Tuple[int, int, int]
    mask: np.ndarray                          # boolean HxW mask
    name: Optional[str] = None

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    def centroid(self) -> Tuple[float, float]:
        ys, xs = np.where(self.mask)
        return float(ys.mean()), float(xs.mean())

    def bbox(self) -> Tuple[int, int, int, int]:
        ys, xs = np.where(self.mask)
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


@dataclass
class ModelingContext:
    """Container for a whole modeling session."""

    image: np.ndarray                         # RGBA HxWx4 uint8
    scale_bar: ScaleBar
    modeling_mask: np.ndarray                 # boolean HxW
    regions: List[Region] = field(default_factory=list)

    def named_regions(self) -> List[Region]:
        return [r for r in self.regions if r.name]

    def total_modeling_pixels(self) -> int:
        return int(self.modeling_mask.sum())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Load a .tif and return an RGBA uint8 HxWx4 array."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.array(img)


# ---------------------------------------------------------------------------
# Scale bar detection
# ---------------------------------------------------------------------------

def detect_scale_bar(
    image: np.ndarray,
    *,
    search_frac_y: float = 0.7,
    search_frac_x: float = 0.5,
) -> ScaleBar:
    """Find the scale bar in the bottom-left of an RGBA image.

    The image background is transparent (alpha=0); the scale bar and its
    caption are drawn with opaque black pixels. The horizontal bar is
    identified as the image row inside the search window that has the
    longest run of opaque-black pixels.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("expected an RGBA image")

    h, w = image.shape[:2]
    rgb = image[..., :3]
    alpha = image[..., 3]

    opaque_black = (alpha > 0) & np.all(rgb == 0, axis=2)

    y_start = int(h * search_frac_y)
    x_end = int(w * search_frac_x)
    window = opaque_black[y_start:, :x_end]
    if not window.any():
        raise RuntimeError("no opaque black pixels found in bottom-left; "
                           "is this the right image?")

    ys, xs = np.where(window)
    y0 = y_start + ys.min()
    y1 = y_start + ys.max()
    x0 = xs.min()
    x1 = xs.max()

    bar_row, bar_x0, bar_x1, bar_len = _longest_black_run(opaque_black, y0, y1 + 1, x0, x1 + 1)

    return ScaleBar(
        bbox=(int(y0), int(y1), int(x0), int(x1)),
        bar_row=int(bar_row),
        bar_x0=int(bar_x0),
        bar_x1=int(bar_x1),
        pixel_length=int(bar_len),
    )


def _longest_black_run(
    mask: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> Tuple[int, int, int, int]:
    """Return (row, x_start, x_end, length) of the longest True run in mask[y0:y1, x0:x1]."""
    best = (y0, x0, x0, 0)
    for r in range(y0, y1):
        row = mask[r, x0:x1]
        if not row.any():
            continue
        # find longest contiguous True run
        idx = np.flatnonzero(row)
        # split into runs where consecutive indices
        splits = np.where(np.diff(idx) != 1)[0]
        starts = np.r_[idx[0], idx[splits + 1]] if splits.size else np.r_[idx[0]]
        ends = np.r_[idx[splits], idx[-1]] if splits.size else np.r_[idx[-1]]
        lengths = ends - starts + 1
        k = lengths.argmax()
        if lengths[k] > best[3]:
            best = (r, x0 + int(starts[k]), x0 + int(ends[k]), int(lengths[k]))
    return best


def set_physical_length(scale_bar: ScaleBar, length: float, unit: str = "nm") -> ScaleBar:
    """Attach a physical length (e.g. 100 nm) to the detected pixel length."""
    scale_bar.physical_length = float(length)
    scale_bar.unit = unit
    return scale_bar


# ---------------------------------------------------------------------------
# Modeling region
# ---------------------------------------------------------------------------

def build_modeling_mask(image: np.ndarray, scale_bar: ScaleBar, pad: int = 2) -> np.ndarray:
    """Return a boolean mask of pixels that are (a) opaque and (b) outside the
    scale-bar bounding box.
    """
    alpha = image[..., 3]
    opaque = alpha > 0
    y0, y1, x0, x1 = scale_bar.bbox
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0] - 1, y1 + pad)
    x1 = min(image.shape[1] - 1, x1 + pad)
    mask = opaque.copy()
    mask[y0:y1 + 1, x0:x1 + 1] = False
    return mask


# ---------------------------------------------------------------------------
# Colour-based segmentation
# ---------------------------------------------------------------------------

def segment_by_color(
    image: np.ndarray,
    modeling_mask: np.ndarray,
    *,
    min_pixels: int = 50,
) -> List[Region]:
    """Group pixels in the modeling region by exact RGB value.

    Works well on schematic images with a small palette. For photographs a
    clustering step would be needed instead.
    """
    rgb = image[..., :3]
    flat = rgb[modeling_mask]
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)

    regions: List[Region] = []
    for i, k in enumerate(order):
        color = tuple(int(c) for c in uniq[k])
        if counts[k] < min_pixels:
            continue
        color_mask = np.all(rgb == np.array(color), axis=2) & modeling_mask
        regions.append(Region(index=i, color=color, mask=color_mask))
    return regions


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(
    ctx: ModelingContext,
    out_path: str | Path,
    *,
    font_path: Optional[str] = None,
    bar_margin: int = 10,
) -> Path:
    """Render a PNG that shows the named regions with labels plus the scale bar."""
    from PIL import ImageDraw, ImageFont

    base = Image.fromarray(ctx.image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_font(font_path, size=14)

    for region in ctx.named_regions():
        cy, cx = region.centroid()
        text = region.name or f"region {region.index}"
        _draw_label(draw, text, (cx, cy), font)

    # scale bar: redraw for clarity and annotate the physical length
    sb = ctx.scale_bar
    y = sb.bar_row
    x0, x1 = sb.bar_x0, sb.bar_x1
    draw.rectangle([x0, y - 2, x1, y + 2], fill=(255, 255, 255, 255))
    if sb.physical_length is not None:
        txt = f"{sb.physical_length:g} {sb.unit}"
        _draw_label(draw, txt, ((x0 + x1) / 2, y - 12), font, anchor="mb")

    out = Image.alpha_composite(base, overlay)
    out_path = Path(out_path)
    out.save(out_path)
    return out_path


def _load_font(font_path: Optional[str], size: int):
    from PIL import ImageFont

    candidates = [font_path] if font_path else []
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def _draw_label(draw, text: str, xy, font, anchor: str = "mm"):
    x, y = xy
    # simple outlined text for legibility on any background
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0, 255), font=font, anchor=anchor)
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font, anchor=anchor)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(ctx: ModelingContext) -> Dict[str, Dict]:
    """Return a dict with per-region pixel count, area in physical units,
    centroid and bounding box.
    """
    sb = ctx.scale_bar
    scale = sb.scale if sb.physical_length is not None else None
    unit = sb.unit

    out: Dict[str, Dict] = {}
    total = ctx.total_modeling_pixels()
    for r in ctx.regions:
        key = r.name or f"region_{r.index}"
        px = r.pixel_count
        info: Dict = {
            "color": r.color,
            "pixel_count": px,
            "fraction_of_modeling": px / total if total else 0.0,
            "centroid_px": r.centroid(),
            "bbox_px": r.bbox(),
        }
        if scale is not None:
            info["area"] = px * scale * scale
            info["area_unit"] = f"{unit}^2"
        out[key] = info
    return out

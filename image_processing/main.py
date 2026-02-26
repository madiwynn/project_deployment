import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps, ImageDraw

# Optional OCR
try:
    import pytesseract  # type: ignore
    HAS_TESS = True
except Exception:
    HAS_TESS = False

# Optional: OpenCV for better edges
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


@dataclass
class Config:
    out_dir: str = "image_processing/out"
    dot_spacing: int = 10
    dot_radius_max: float = 5.0
    sample_box: int = 7
    max_dim: int = 2400
    svg_background: str = "#ffffff"
    png_background: Tuple[int, int, int] = (255, 255, 255)
    palette_k: int = 6
    ocr_enabled: bool = True
    ocr_lang: str = "eng"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path: str, max_dim: int = 0) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def kmeans_palette(pixels: np.ndarray, k: int, iters: int = 12, seed: int = 7) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    N = pixels.shape[0]
    if N == 0:
        return [(0, 0, 0)]

    sample_n = min(N, 6000)
    idx = rng.choice(N, size=sample_n, replace=False)
    X = pixels[idx].astype(np.float32)

    centers = X[rng.choice(sample_n, size=min(k, sample_n), replace=False)]

    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        new_centers = []
        for ci in range(centers.shape[0]):
            pts = X[labels == ci]
            if len(pts) == 0:
                new_centers.append(centers[ci])
            else:
                new_centers.append(pts.mean(axis=0))
        centers = np.vstack(new_centers)

    centers = np.clip(centers, 0, 255).astype(np.uint8)

    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    counts = np.bincount(labels, minlength=centers.shape[0])
    order = np.argsort(-counts)
    centers = centers[order]

    return [tuple(map(int, c)) for c in centers]


def canny_edge_density(img_rgb: np.ndarray) -> float:
    h, w, _ = img_rgb.shape
    if HAS_CV2:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float((edges > 0).mean())

    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > (mag.mean() + mag.std())).mean())


def sky_heuristic(img_rgb: np.ndarray) -> float:
    h, w, _ = img_rgb.shape
    top = img_rgb[: h // 2]
    r, g, b = top[..., 0].astype(np.float32), top[..., 1].astype(np.float32), top[..., 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    skyish = (b > r + 15) & (b > g + 10) & (bright > 60)
    return float(skyish.mean())


def ocr_text(img: Image.Image, lang: str) -> Optional[str]:
    if not HAS_TESS:
        return None
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    txt = pytesseract.image_to_string(gray, lang=lang)
    txt = "\n".join([line.rstrip() for line in txt.splitlines() if line.strip()])
    return txt or None


def dot_reconstruct_png(img: Image.Image, cfg: Config, out_path: str) -> None:
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)
    out = Image.new("RGB", (w, h), cfg.png_background)
    draw = ImageDraw.Draw(out)

    half = cfg.sample_box // 2
    lum = luminance(arr.astype(np.float32))
    lum_min, lum_max = float(lum.min()), float(lum.max())
    denom = max(1e-6, (lum_max - lum_min))

    for y in range(0, h, cfg.dot_spacing):
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        for x in range(0, w, cfg.dot_spacing):
            x0 = max(0, x - half)
            x1 = min(w, x + half + 1)

            patch = arr[y0:y1, x0:x1]
            rgb = patch.reshape(-1, 3).mean(axis=0)
            L = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
            t = 1.0 - ((L - lum_min) / denom)
            r = float(cfg.dot_radius_max) * float(t)

            if r < 0.4:
                continue

            fill = tuple(map(int, rgb))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    out.save(out_path)


def dot_reconstruct_svg(img: Image.Image, cfg: Config, out_path: str) -> None:
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)

    half = cfg.sample_box // 2
    lum = luminance(arr.astype(np.float32))
    lum_min, lum_max = float(lum.min()), float(lum.max())
    denom = max(1e-6, (lum_max - lum_min))

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append(f'<rect width="100%" height="100%" fill="{cfg.svg_background}"/>')

    for y in range(0, h, cfg.dot_spacing):
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        for x in range(0, w, cfg.dot_spacing):
            x0 = max(0, x - half)
            x1 = min(w, x + half + 1)

            patch = arr[y0:y1, x0:x1]
            rgb = patch.reshape(-1, 3).mean(axis=0)
            L = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
            t = 1.0 - ((L - lum_min) / denom)
            r = float(cfg.dot_radius_max) * float(t)
            if r < 0.4:
                continue

            parts.append(
                f'<circle cx="{x}" cy="{y}" r="{r:.3f}" fill="{rgb_to_hex(tuple(map(int, rgb)))}" />'
            )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


@dataclass
class ImageDigest:
    filename: str
    width: int
    height: int
    aspect_ratio: float
    mean_brightness: float
    contrast: float
    edge_density: float
    sky_score: float
    dominant_palette: List[str]
    ocr_text: Optional[str]


def build_digest(img: Image.Image, cfg: Config, filename: str) -> ImageDigest:
    arr = np.array(img, dtype=np.uint8)
    w, h = img.size
    ar = w / h

    lum = luminance(arr.astype(np.float32))
    mean_b = float(lum.mean())
    contrast = float(lum.std())

    edge = canny_edge_density(arr)
    sky = sky_heuristic(arr)

    pixels = arr.reshape(-1, 3)
    palette = kmeans_palette(pixels, k=cfg.palette_k)
    palette_hex = [rgb_to_hex(c) for c in palette]

    text = None
    if cfg.ocr_enabled:
        text = ocr_text(img, cfg.ocr_lang)

    return ImageDigest(
        filename=filename,
        width=w,
        height=h,
        aspect_ratio=float(ar),
        mean_brightness=mean_b,
        contrast=contrast,
        edge_density=edge,
        sky_score=sky,
        dominant_palette=palette_hex,
        ocr_text=text
    )


def process_images(image_paths: List[str], cfg: Config) -> None:
    ensure_dir(cfg.out_dir)
    digests: List[Dict] = []

    for p in image_paths:
        img = load_image(p, max_dim=cfg.max_dim)
        base = os.path.splitext(os.path.basename(p))[0]

        svg_path = os.path.join(cfg.out_dir, f"{base}.dots.svg")
        png_path = os.path.join(cfg.out_dir, f"{base}.dots.png")

        dot_reconstruct_svg(img, cfg, svg_path)
        dot_reconstruct_png(img, cfg, png_path)

        d = build_digest(img, cfg, os.path.basename(p))
        digests.append(asdict(d))

    with open(os.path.join(cfg.out_dir, "digest.json"), "w", encoding="utf-8") as f:
        json.dump(digests, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    IMAGE_PATHS = [
        # Example:
        # "image_processing/images/photo1.jpg",
    ]

    cfg = Config()

    if not IMAGE_PATHS:
        raise SystemExit("Set IMAGE_PATHS in image_processing/main.py")

    process_images(IMAGE_PATHS, cfg)
    print("Done. Outputs in image_processing/out")

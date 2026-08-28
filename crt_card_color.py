#!/usr/bin/env python3
"""Generate a colour CRT-style profile card PNG for a GitHub profile README.

Usage:  python3 crt_card_color.py <avatar.png> [out.png]
Edit CONFIG / THEME below.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CONFIG = {
    "prompt": "lyly@space$",
    "lines": [
        ("Name",   "lyly"),
        ("Focus",  "AI Agents / LLM Apps"),
        ("OS",     "macOS, Linux"),
        ("Editor", "VS Code"),
        ("Skills", "Agent Dev, Web Scraping,\nData Analysis, Backend"),
    ],
    "swatches": [],          # empty -> the colour bar is skipped
}

THEME = {
    "screen":  (12, 14, 26),      # CRT background
    "panel":   (250, 245, 238),   # backdrop behind the avatar
    "label":   (232, 122, 66),    # "Name:" etc
    "value":   (240, 234, 224),
    "prompt":  (108, 168, 255),
    "rail":    (196, 206, 232),   # bar under the swatches
    "bezel":   (20, 20, 24),
}

METEOR_CORE = (255, 252, 240)
METEOR_TAIL = (86, 132, 226)

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 34
LINE_H = 60        # vertical spacing between rows
LABEL_GAP = 5     # blank chars between 'Name:' and its value

PIXEL = 4                     # size of one dithered pixel block
AV_COLS, AV_ROWS = 190, 172   # avatar resolution in blocks (match your art's aspect)
LEVELS = 6                    # quantisation steps per RGB channel
LINEART = True                # keep thin strokes when downscaling
PAD = 46

BAYER = np.array([
    [0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5],
], dtype=float) / 16.0
LUM = np.array([0.299, 0.587, 0.114])


def _crop_to_aspect(im, tw, th):
    w, h = im.size
    if w / h > tw / th:
        nw = int(h * tw / th)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w * th / tw)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def dither(path, cols, rows, levels=LEVELS):
    """Downscale an image and ordered-dither it to a small RGB palette."""
    im = _crop_to_aspect(Image.open(path).convert("RGB"), cols, rows)

    if LINEART:
        # sample at 2x then keep the DARKEST pixel of each 2x2 block, so thin
        # strokes survive instead of being averaged away into the paper.
        a = np.asarray(im.resize((cols * 2, rows * 2), Image.LANCZOS), dtype=float)
        a = a.reshape(rows, 2, cols, 2, 3).transpose(0, 2, 1, 3, 4).reshape(rows, cols, 4, 3)
        k = (a @ LUM).argmin(axis=2)
        a = np.take_along_axis(a, k[..., None, None], axis=2)[:, :, 0, :]
    else:
        a = np.asarray(im.resize((cols, rows), Image.LANCZOS), dtype=float)

    a = np.clip((a / 255.0 - 0.5) * 1.12 + 0.5, 0, 1)                 # contrast
    q = a * (levels - 1)
    base = np.floor(q)
    thr = np.tile(BAYER, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols][..., None]
    idx = np.clip(base + ((q - base) > thr), 0, levels - 1)
    rgb = (idx / (levels - 1) * 255).astype(np.uint8)

    return Image.fromarray(rgb).resize((cols * PIXEL, rows * PIXEL), Image.NEAREST)


def draw_sky(screen, box, seed=11):
    """A few faint stars and one meteor streaking down-left across `box`."""
    x0, y0, x1, y1 = box
    sky = Image.new("RGBA", screen.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(sky)
    rng = np.random.default_rng(seed)

    for _ in range(70):
        sx = rng.uniform(x0, x1)
        sy = rng.uniform(y0, y1)
        r = rng.uniform(1.0, 2.6)
        a = int(rng.uniform(40, 170))
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(214, 226, 255, a))

    # trail runs from the upper right down to the lower left
    hx, hy = x0 + (x1 - x0) * 0.20, y0 + (y1 - y0) * 0.82   # head
    tx_, ty_ = x0 + (x1 - x0) * 0.94, y0 + (y1 - y0) * 0.12  # tail
    core = np.array(METEOR_CORE, dtype=float)
    tail = np.array(METEOR_TAIL, dtype=float)

    steps = 300
    # two passes: a wide soft halo, then a thin bright core on top of it
    for r_max, a_max, blur in ((17.0, 60, 7), (5.2, 240, 0)):
        layer = Image.new("RGBA", screen.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for i in range(steps):
            t = i / (steps - 1)             # 0 = far tail, 1 = head
            x = tx_ + (hx - tx_) * t
            y = ty_ + (hy - ty_) * t
            r = max(0.5, r_max * t ** 2.1)
            a = int(a_max * t ** 1.8)
            c = tuple(int(v) for v in (tail + (core - tail) * t ** 1.4))
            ld.ellipse([x - r, y - r, x + r, y + r], fill=c + (a,))
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        sky = Image.alpha_composite(sky, layer)

    halo = Image.new("RGBA", screen.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    for i in range(34, 0, -1):
        r = i
        a = int(220 * (1 - i / 34) ** 2.0) + (200 if i <= 4 else 0)
        hd.ellipse([hx - r, hy - r, hx + r, hy + r],
                   fill=METEOR_CORE + (min(a, 255),))
    sky = Image.alpha_composite(sky, halo.filter(ImageFilter.GaussianBlur(4)))

    return Image.alpha_composite(screen.convert("RGBA"), sky).convert("RGB")


def barrel(im, k=0.055):
    """Bulge the image outward like a curved CRT tube."""
    a = np.asarray(im, dtype=np.uint8)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = (xx / (w - 1)) * 2 - 1, (yy / (h - 1)) * 2 - 1
    f = 1 - k * (nx ** 2 + ny ** 2)
    sx = np.clip(((nx * f + 1) / 2 * (w - 1)).round(), 0, w - 1).astype(int)
    sy = np.clip(((ny * f + 1) / 2 * (h - 1)).round(), 0, h - 1).astype(int)
    return Image.fromarray(a[sy, sx])


def crt(im):
    """Chromatic fringing + scanlines + aperture grille + vignette + bloom."""
    a = np.asarray(im, dtype=float)
    h, w = a.shape[:2]

    a[:, 1:, 0] = a[:, :-1, 0]          # red lags one column
    a[:, :-1, 2] = a[:, 1:, 2]          # blue leads one column

    a *= np.where((np.arange(h) % 4) < 2, 1.0, 0.74)[:, None, None]
    pat = np.array([[1.0, 0.92, 0.96], [0.96, 1.0, 0.92], [0.92, 0.96, 1.0]])
    grille = np.tile(pat, (w // 3 + 1, 1))[:w]
    a *= grille[None, :, :]

    ny, nx = np.mgrid[0:h, 0:w].astype(float)
    r = np.hypot(nx / (w - 1) * 2 - 1, ny / (h - 1) * 2 - 1)
    a *= np.clip(1.05 - 0.33 * r ** 2.4, 0, 1)[..., None]

    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    glow = im.filter(ImageFilter.GaussianBlur(9))
    b = np.asarray(im, float) + np.asarray(glow, float) * 0.34
    return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))


def build(avatar_path, out_path):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    av = dither(avatar_path, AV_COLS, AV_ROWS)

    label_w = max(len(k) for k, _ in CONFIG["lines"]) + LABEL_GAP

    # size the text column to its widest line so nothing runs off the tube
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lab_px = probe.textlength(" " * label_w, font=font)
    val_px = max(probe.textlength(part, font=font)
                 for _, v in CONFIG["lines"] for part in v.split("\n"))
    W = PAD * 2 + av.width + 44 + int(lab_px + val_px) + 96
    H = PAD * 2 + av.height + 124

    screen = Image.new("RGB", (W, H), THEME["screen"])
    d = ImageDraw.Draw(screen)

    ax, ay = PAD, PAD
    d.rectangle([ax - 8, ay - 8, ax + av.width + 7, ay + av.height + 7], fill=THEME["panel"])
    screen.paste(av, (ax, ay))

    tx, ty = ax + av.width + 44, PAD + 4
    for key, val in CONFIG["lines"]:
        d.text((tx, ty), f"{key}:", font=font, fill=THEME["label"])
        vx = tx + int(d.textlength(" " * label_w, font=font))
        for i, part in enumerate(val.split("\n")):
            d.text((vx, ty), part, font=font, fill=THEME["value"])
            ty += LINE_H

    if CONFIG["swatches"]:
        sw, sh = 62, 46
        sx, sy = tx, ty + 34
        for i, hexc in enumerate(CONFIG["swatches"]):
            d.rectangle([sx + i * sw, sy, sx + (i + 1) * sw - 1, sy + sh], fill=hexc)
        d.rectangle([sx - 10, sy + sh, sx + len(CONFIG["swatches"]) * sw + 10, sy + sh + 18],
                    fill=THEME["rail"])

    screen = draw_sky(screen, (tx - 10, ty + 70, W - PAD - 20, H - PAD - 20))
    d = ImageDraw.Draw(screen)

    py = H - PAD - FONT_SIZE - 30
    d.text((PAD + 40, py), CONFIG["prompt"], font=font, fill=THEME["prompt"])
    cx = PAD + 40 + int(d.textlength(CONFIG["prompt"] + " ", font=font))
    d.rectangle([cx, py - 2, cx + 16, py + FONT_SIZE + 2], fill=THEME["prompt"])

    screen = crt(barrel(screen))

    B = 34
    card = Image.new("RGB", (W + B * 2, H + B * 2), THEME["bezel"])
    mask = Image.new("L", screen.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=26, fill=255)
    card.paste(screen, (B, B), mask)
    card.save(out_path)
    print(f"wrote {out_path}  ({card.width}x{card.height})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "card.png")

#!/usr/bin/env python3
"""Render a CRT-style profile card for a GitHub profile README.

    python3 crt_card_color.py <avatar.jpg> card.png    # single still frame
    python3 crt_card_color.py <avatar.jpg> card.gif    # animated loop

Animated: a meteor streaks past, the cursor blinks, the scanlines drift and
the tube flickers. The loop is built to be seamless.
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
    "screen":  (12, 14, 26),
    "panel":   (250, 245, 238),
    "label":   (232, 122, 66),
    "value":   (240, 234, 224),
    "prompt":  (108, 168, 255),
    "rail":    (196, 206, 232),
    "bezel":   (20, 20, 24),
}
METEOR_CORE = (255, 252, 240)
METEOR_TAIL = (86, 132, 226)

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 34
LINE_H = 60
LABEL_GAP = 5

PIXEL = 4
AV_COLS, AV_ROWS = 190, 172
LEVELS = 6
LINEART = True
PAD = 46

OUT_W = 900              # final pixel width of the card
FRAMES = 48              # multiple of 4 keeps the scanline drift seamless
FRAME_MS = 60
METEOR_FRAMES = 16       # how long one fly-past lasts
N_STARS = 70

# Global per-frame changes (drifting scanlines, tube flicker) touch every pixel
# and defeat the GIF's frame-to-frame compression. Off = far smaller file.
SCAN_DRIFT = False
FLICKER_AMT = 0.0

BAYER = np.array([
    [0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5],
], dtype=float) / 16.0
LUM = np.array([0.299, 0.587, 0.114])


# ---------------------------------------------------------------- avatar ---

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

    a = np.clip((a / 255.0 - 0.5) * 1.12 + 0.5, 0, 1)
    q = a * (levels - 1)
    base = np.floor(q)
    thr = np.tile(BAYER, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols][..., None]
    idx = np.clip(base + ((q - base) > thr), 0, levels - 1)
    rgb = (idx / (levels - 1) * 255).astype(np.uint8)

    return Image.fromarray(rgb).resize((cols * PIXEL, rows * PIXEL), Image.NEAREST)


# ------------------------------------------------------------------- sky ---

def star_field(box, n=N_STARS, seed=11):
    """Fixed star positions + a per-star twinkle phase."""
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    return [(rng.uniform(x0, x1), rng.uniform(y0, y1),
             rng.uniform(1.0, 2.6), rng.uniform(40, 170), rng.uniform(0, 1))
            for _ in range(n)]


def draw_sky(screen, box, stars, t, meteor_p):
    """Stars twinkle; if meteor_p is not None a meteor is mid-flight at that
    progress (0 = just entering, 1 = leaving)."""
    sky = Image.new("RGBA", screen.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(sky)

    for sx, sy, r, a, ph in stars:
        tw = 0.55 + 0.45 * np.sin(2 * np.pi * (2 * t + ph))
        d.ellipse([sx - r, sy - r, sx + r, sy + r],
                  fill=(214, 226, 255, int(a * tw)))

    if meteor_p is not None:
        x0, y0, x1, y1 = box
        # the flight path: enters top-right, exits bottom-left
        ax, ay = x0 + (x1 - x0) * 1.05, y0 + (y1 - y0) * 0.02
        bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.95
        p = -0.15 + meteor_p * 1.30                 # head travels off both ends
        hx, hy = ax + (bx - ax) * p, ay + (by - ay) * p
        tl = 0.42                                   # trail length along the path
        tx_, ty_ = ax + (bx - ax) * (p - tl), ay + (by - ay) * (p - tl)

        fade = min(1.0, meteor_p / 0.18, (1.0 - meteor_p) / 0.22)
        fade = max(0.0, fade)
        core = np.array(METEOR_CORE, dtype=float)
        tail = np.array(METEOR_TAIL, dtype=float)

        steps = 300
        for r_max, a_max, blur in ((17.0, 60, 7), (5.2, 240, 0)):
            layer = Image.new("RGBA", screen.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            for i in range(steps):
                s = i / (steps - 1)                 # 0 = far tail, 1 = head
                x = tx_ + (hx - tx_) * s
                y = ty_ + (hy - ty_) * s
                r = max(0.5, r_max * s ** 2.1)
                a = int(a_max * s ** 1.8 * fade)
                if a <= 0:
                    continue
                c = tuple(int(v) for v in (tail + (core - tail) * s ** 1.4))
                ld.ellipse([x - r, y - r, x + r, y + r], fill=c + (a,))
            if blur:
                layer = layer.filter(ImageFilter.GaussianBlur(blur))
            sky = Image.alpha_composite(sky, layer)

        halo = Image.new("RGBA", screen.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo)
        for i in range(34, 0, -1):
            a = int(220 * (1 - i / 34) ** 2.0) + (200 if i <= 4 else 0)
            hd.ellipse([hx - i, hy - i, hx + i, hy + i],
                       fill=METEOR_CORE + (min(int(a * fade), 255),))
        sky = Image.alpha_composite(sky, halo.filter(ImageFilter.GaussianBlur(4)))

    return Image.alpha_composite(screen.convert("RGBA"), sky).convert("RGB")


# ------------------------------------------------------------------- CRT ---

def barrel_maps(w, h, k=0.055):
    """Precompute the tube-curvature sample indices once."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx, ny = (xx / (w - 1)) * 2 - 1, (yy / (h - 1)) * 2 - 1
    f = 1 - k * (nx ** 2 + ny ** 2)
    sx = np.clip(((nx * f + 1) / 2 * (w - 1)).round(), 0, w - 1).astype(int)
    sy = np.clip(((ny * f + 1) / 2 * (h - 1)).round(), 0, h - 1).astype(int)
    return sy, sx


def crt(im, scan_offset=0, flicker=1.0, blur=5):
    """Chromatic fringing + drifting scanlines + grille + vignette + bloom."""
    a = np.asarray(im, dtype=float)
    h, w = a.shape[:2]

    a[:, 1:, 0] = a[:, :-1, 0]
    a[:, :-1, 2] = a[:, 1:, 2]

    a *= np.where(((np.arange(h) + scan_offset) % 4) < 2, 1.0, 0.74)[:, None, None]
    pat = np.array([[1.0, 0.92, 0.96], [0.96, 1.0, 0.92], [0.92, 0.96, 1.0]])
    a *= np.tile(pat, (w // 3 + 1, 1))[:w][None, :, :]

    ny, nx = np.mgrid[0:h, 0:w].astype(float)
    r = np.hypot(nx / (w - 1) * 2 - 1, ny / (h - 1) * 2 - 1)
    a *= np.clip(1.05 - 0.33 * r ** 2.4, 0, 1)[..., None] * flicker

    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    glow = im.filter(ImageFilter.GaussianBlur(blur))
    b = np.asarray(im, float) + np.asarray(glow, float) * 0.34
    return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------- build ---

def build_base(avatar_path):
    """Everything that never moves: panel, avatar, labels, prompt text."""
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    av = dither(avatar_path, AV_COLS, AV_ROWS)
    label_w = max(len(k) for k, _ in CONFIG["lines"]) + LABEL_GAP

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
        vx = tx + int(lab_px)
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

    py = H - PAD - FONT_SIZE - 30
    d.text((PAD + 40, py), CONFIG["prompt"], font=font, fill=THEME["prompt"])
    cx = PAD + 40 + int(d.textlength(CONFIG["prompt"] + " ", font=font))

    box = (tx - 10, ty + 70, W - PAD - 20, H - PAD - 20)
    return screen, {
        "W": W, "H": H, "box": box, "stars": star_field(box),
        "cursor": (cx, py - 2, cx + 16, py + FONT_SIZE + 2),
        "maps": barrel_maps(W, H),
        "scale": OUT_W / (W + 68),
    }


def render_frame(base, L, i, animated):
    t = i / FRAMES
    meteor_p = i / (METEOR_FRAMES - 1) if animated and i < METEOR_FRAMES else None
    if not animated:
        meteor_p = 0.55                       # a nice mid-flight pose for the still

    screen = draw_sky(base.copy(), L["box"], L["stars"], t, meteor_p)

    if not animated or (i // 8) % 2 == 0:     # blinking block cursor
        ImageDraw.Draw(screen).rectangle(L["cursor"], fill=THEME["prompt"])

    sy, sx = L["maps"]
    screen = Image.fromarray(np.asarray(screen, dtype=np.uint8)[sy, sx])

    ow = int(L["W"] * L["scale"])
    oh = int(L["H"] * L["scale"])
    screen = screen.resize((ow, oh), Image.LANCZOS)

    flicker = 1.0 + FLICKER_AMT * np.sin(2 * np.pi * 3 * t) if animated else 1.0
    screen = crt(screen, scan_offset=(i % 4) if (animated and SCAN_DRIFT) else 0,
                 flicker=flicker)

    B = int(34 * L["scale"])
    card = Image.new("RGB", (ow + B * 2, oh + B * 2), THEME["bezel"])
    mask = Image.new("L", screen.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, ow - 1, oh - 1],
                                           radius=int(26 * L["scale"]), fill=255)
    card.paste(screen, (B, B), mask)
    return card


def build(avatar_path, out_path):
    base, L = build_base(avatar_path)

    if not out_path.lower().endswith(".gif"):
        render_frame(base, L, 0, animated=False).save(out_path)
        print(f"wrote {out_path}")
        return

    frames = []
    for i in range(FRAMES):
        frames.append(render_frame(base, L, i, animated=True))
        print(f"  frame {i + 1}/{FRAMES}", end="\r", flush=True)
    print()

    # one shared palette, sampled across the loop, so colours don't crawl
    picks = [frames[j] for j in (0, 6, 10, 14, 20, 34)]
    strip = Image.new("RGB", (picks[0].width * len(picks), picks[0].height))
    for j, f in enumerate(picks):
        strip.paste(f, (j * f.width, 0))
    pal = strip.quantize(colors=192, method=Image.MEDIANCUT)

    qs = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    qs[0].save(out_path, save_all=True, append_images=qs[1:],
               duration=FRAME_MS, loop=0, optimize=True)
    print(f"wrote {out_path}  ({qs[0].width}x{qs[0].height}, {FRAMES} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "card.png")

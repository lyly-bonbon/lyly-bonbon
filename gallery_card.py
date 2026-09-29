#!/usr/bin/env python3
"""Gallery-style profile card: the portrait hangs in a gilt frame beside a
museum wall label. Animated: dust drifts through the spotlight, cursor blinks.

    python3 gallery_card.py <painting> assets/card.gif   # animated loop
    python3 gallery_card.py <painting> card.png          # single still
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

TITLE = "lyly"
KICKER = "SELF-PORTRAIT  ·  2026"
MEDIUM = "Oil on canvas, Python on macOS"
FIELDS = [("Focus", "AI Agents / LLM Apps"),
          ("Skills", "Agent Dev, Web Scraping,\nData Analysis, Backend"),
          ("Editor", "VS Code"),
          ("OS", "macOS, Linux")]
PROMPT = "lyly@space$"
SEAL = ("ly", "ly")

SUP = "/System/Library/Fonts/Supplemental/"
MONO = "/System/Library/Fonts/Menlo.ttc"

OUT_W, OUT_H = 900, 560
S = 2                                  # draw at 2x, downscale once
W, H = OUT_W * S, OUT_H * S

WALL_LIT, WALL_DARK = (44, 34, 26), (12, 9, 7)
GOLD, IVORY, MUTED = (200, 164, 98), (240, 229, 206), (172, 152, 120)
VERMILION = (176, 36, 28)

SKIN_LIFT = 9            # L* added to skin tones (0 = leave the painting as is)
SKIN_DEYELLOW = 0.45     # share of the skin's yellow (b*) taken out

# painting + frame geometry, in 2x px
P = 860
FW, LINER = 44, 8
OUT = P + 2 * (FW + LINER)
FX, FY = 90, (H - OUT) // 2
PX, PY = FX + FW + LINER, FY + FW + LINER
TX = FX + OUT + 110
LIGHT = (FX + OUT / 2 + 120, -H * 0.1)  # spotlight source, above the frame

FRAMES = 64              # 8 cursor half-blinks -> seamless loop
FRAME_MS = 65
BLINK = 8
N_MOTES = 46


def F(name, size, idx=0):
    return ImageFont.truetype(SUP + name, size, index=idx)


def light_falloff(xx, yy):
    """0 at the lamp, 1 where the spotlight has run out (2x coordinates)."""
    return np.clip(np.hypot((xx - LIGHT[0]) / (W * 0.62), (yy - LIGHT[1]) / (H * 1.05)), 0, 1)


def wall():
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    t = light_falloff(xx, yy)[..., None] ** 1.4
    a = np.array(WALL_LIT) * (1 - t) + np.array(WALL_DARK) * t
    a += np.random.default_rng(3).normal(0, 2.0, (H, W, 1))     # plaster grain
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


GILT = [(0.00, (54, 38, 14)), (0.12, (150, 112, 44)), (0.28, (238, 208, 132)),
        (0.42, (158, 120, 48)), (0.58, (216, 180, 100)), (0.78, (120, 86, 30)),
        (0.92, (70, 50, 18)), (1.00, (38, 27, 10))]


def gilt(t):
    for (t0, c0), (t1, c1) in zip(GILT, GILT[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
    return GILT[-1][1]


def gilt_frame(img):
    """Drop shadow, carved gold moulding lit from the upper left, dark liner."""
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rectangle([FX + 12, FY + 26, FX + OUT + 12, FY + OUT + 26], fill=200)
    img = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, m.filter(ImageFilter.GaussianBlur(30)))

    d = ImageDraw.Draw(img)
    for i in range(FW):
        d.rectangle([FX + i, FY + i, FX + OUT - i, FY + OUT - i], outline=gilt(i / (FW - 1)))
    d.rectangle([PX - LINER, PY - LINER, PX + P + LINER - 1, PY + P + LINER - 1], fill=(24, 17, 10))

    a = np.asarray(img, dtype=float)
    yy, xx = np.mgrid[FY:FY + OUT + 1, FX:FX + OUT + 1].astype(float)
    a[FY:FY + OUT + 1, FX:FX + OUT + 1] *= (1.14 - 0.34 * ((xx - FX) + (yy - FY)) / (2 * OUT))[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def seal(size):
    """Vermilion seal with the letters carved out (阴刻), stamped slightly askew."""
    s = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(s)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.07), fill=VERMILION + (255,))
    f = F("Bodoni 72.ttc", int(size * 0.40), 2)
    for i, t in enumerate(SEAL):
        w = d.textlength(t, font=f)
        d.text(((size - w) / 2, size * 0.02 + i * size * 0.44), t, font=f, fill=(0, 0, 0, 0))
    a = np.asarray(s).copy()
    g = np.random.default_rng(5).uniform(0.45, 1.0, (size, size))
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))) / 255
    a[..., 3] = (a[..., 3] * np.clip(g * 1.3, 0, 1)).astype(np.uint8)       # uneven ink
    return Image.fromarray(a).rotate(-4, resample=Image.BICUBIC, expand=True)


def tracked(d, xy, text, font, fill, track):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.einsum("...j,ij->...i", lin, _M) / _WHITE    # einsum: avoids spurious BLAS FP warnings on macOS
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    lin = np.clip(np.einsum("...j,ij->...i", xyz, np.linalg.inv(_M)), 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def skin_mask(lab):
    """Soft mask of lit skin: mid-to-high lightness, skin hue, moderate chroma.
    The dark ground, navy dress and the saturated red scarf/lips fall outside."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C, h = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    m = (_ramp(L, 30, 42) * _ramp(h, 22, 34) * (1 - _ramp(h, 62, 76))
         * _ramp(C, 6, 12) * (1 - _ramp(C, 40, 52)))
    m = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    return np.asarray(m, dtype=float) / 255


def fair_skin(img):
    """Lighten the skin and take the yellow out of it; everything else untouched."""
    lab = rgb_to_lab(np.asarray(img, dtype=float))
    m = skin_mask(lab)
    lab[..., 0] += SKIN_LIFT * m
    lab[..., 2] *= 1 - SKIN_DEYELLOW * m
    return Image.fromarray(lab_to_rgb(lab).astype(np.uint8))


def build_base(painting_path):
    """Everything that never moves, downscaled to the output size, plus the
    cursor rectangle in output pixels."""
    img = gilt_frame(wall())
    img.paste(fair_skin(Image.open(painting_path).convert("RGB").resize((P, P), Image.LANCZOS)), (PX, PY))
    d = ImageDraw.Draw(img)

    y = FY + 30
    tracked(d, (TX, y), KICKER, F("Baskerville.ttc", 24), GOLD, 5)
    y += 44
    d.text((TX - 6, y), TITLE, font=F("Didot.ttc", 150, 1), fill=IVORY)
    y += 186
    d.text((TX, y), MEDIUM, font=F("Baskerville.ttc", 31, 2), fill=MUTED)
    y += 70
    d.line([TX, y, TX + 110, y], fill=GOLD, width=2)
    y += 40

    f_sc, f_val = F("Bodoni 72 Smallcaps Book.ttf", 27), F("Baskerville.ttc", 33)
    for label, val in FIELDS:
        tracked(d, (TX, y), label, f_sc, GOLD, 3)
        y += 36
        for part in val.split("\n"):
            d.text((TX, y), part, font=f_val, fill=IVORY)
            y += 42
        y += 20

    st = seal(118)
    sy = FY + OUT - st.height - 6
    img.paste(st, (TX - 4, sy), st)
    mono = ImageFont.truetype(MONO, 26)
    px, py = TX + st.width + 26, sy + st.height // 2 - 14
    d.text((px, py), PROMPT, font=mono, fill=MUTED)
    cx = px + d.textlength(PROMPT + " ", font=mono)
    cursor = tuple(v / S for v in (cx, py + 3, cx + 14, py + 31))

    return img.resize((OUT_W, OUT_H), Image.LANCZOS), cursor


def make_motes(seed=7):
    rng = np.random.default_rng(seed)
    return [dict(x=rng.uniform(10, OUT_W - 10), y=rng.uniform(0, OUT_H * 0.95),
                 ax=rng.uniform(4, 14), ay=rng.uniform(6, 18),
                 p1=rng.uniform(0, 2 * np.pi), p2=rng.uniform(0, 2 * np.pi),
                 p3=rng.uniform(0, 1), r=rng.uniform(0.7, 2.0))
            for _ in range(N_MOTES)]


def render_frame(base, cursor, motes, i):
    """Dust floats on small closed loops (so the GIF loops cleanly) and only
    shows where the spotlight reaches; it glints as it turns in the light."""
    t = i / FRAMES
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for m in motes:
        x = m["x"] + m["ax"] * np.sin(2 * np.pi * t + m["p1"])
        y = m["y"] + m["ay"] * np.sin(2 * np.pi * t + m["p2"])
        lit = 1 - light_falloff(x * S, y * S)
        glint = 0.35 + 0.65 * np.sin(np.pi * (t + m["p3"])) ** 2
        a = int(170 * lit * glint)
        if a < 6:
            continue
        r = m["r"]
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 228, 176, a))
    out = Image.alpha_composite(base.convert("RGBA"), layer.filter(ImageFilter.GaussianBlur(0.7))).convert("RGB")
    if (i // BLINK) % 2 == 0:
        ImageDraw.Draw(out).rectangle(cursor, fill=GOLD)
    return out


def _palette_from(pixels, n):
    """Median-cut palette of up to n colours for an (N, 3) pixel array."""
    side = int(np.ceil(np.sqrt(len(pixels))))
    img = Image.fromarray(np.resize(pixels, (side * side, 3)).reshape(side, side, 3).astype(np.uint8))
    return np.array(img.quantize(colors=n, method=Image.Quantize.MEDIANCUT).getpalette()[:n * 3]).reshape(-1, 3)


def save_gif(base, frames, path):
    """Quantise the static base once (with dithering, so the painting and the
    wall gradient stay smooth), then only remap the pixels each frame changes.
    Static pixels keep identical indices frame to frame, which keeps it small."""
    # fixed colour budgets per region, so the gilt frame, the painting and the
    # skin can't starve one another (255 in all; one slot kept for transparency)
    box = tuple(int(v / S) for v in (PX, PY, PX + P, PY + P))
    rgb = np.asarray(base)
    in_art = np.zeros(rgb.shape[:2], bool)
    in_art[box[1]:box[3], box[0]:box[2]] = True
    skin = np.zeros_like(in_art)
    skin[box[1]:box[3], box[0]:box[2]] = skin_mask(rgb_to_lab(rgb[box[1]:box[3], box[0]:box[2]].astype(float))) > 0.4
    budget = np.concatenate([_palette_from(rgb[~in_art], 100),
                             _palette_from(rgb[in_art & ~skin], 95),
                             _palette_from(rgb[skin], 60)])[:255]
    budget = np.concatenate([budget, np.repeat(budget[:1], 256 - len(budget), 0)])   # duplicate pad: never picked
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(budget.astype(np.uint8).flatten().tolist())
    q = base.quantize(palette=pal_img)
    pal_flat = q.getpalette()
    pal = np.array(pal_flat, dtype=float).reshape(-1, 3)
    base_rgb = np.asarray(base, dtype=np.int16)
    base_idx = np.asarray(q)

    out = []
    for f in frames:
        rgb = np.asarray(f, dtype=np.int16)
        changed = np.any(rgb != base_rgb, axis=2)
        idx = base_idx.copy()
        if changed.any():
            px = rgb[changed].astype(float)
            idx[changed] = ((px[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
        im = Image.frombytes("P", base.size, idx.astype(np.uint8).tobytes())
        im.putpalette(pal_flat)
        out.append(im)
    # optimize=True makes unchanged pixels transparent in later frames
    out[0].save(path, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, optimize=True)


def build(painting_path, out_path):
    base, cursor = build_base(painting_path)
    motes = make_motes()
    if not out_path.lower().endswith(".gif"):
        render_frame(base, cursor, motes, 0).save(out_path)
        print(f"wrote {out_path}")
        return
    frames = [render_frame(base, cursor, motes, i) for i in range(FRAMES)]
    save_gif(base, frames, out_path)
    print(f"wrote {out_path}  ({OUT_W}x{OUT_H}, {FRAMES} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "card.png")

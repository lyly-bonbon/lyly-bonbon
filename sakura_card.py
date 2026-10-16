#!/usr/bin/env python3
"""Cherry-blossom profile card: Sakura flies over the night city while petals
drift down past her. The petals painted into the art are lifted out first, so
every petal on screen is a moving one.

    python3 sakura_card.py <art> assets/card.gif   # animated loop
    python3 sakura_card.py <art> card.png          # single still
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

TITLE = "lyly"
TAGLINE = "AI AGENTS / LLM APPS"
FIELDS = [("SKILLS", ["Agent Dev", "Web Scraping", "Data Analysis", "Backend"]),
          ("EDITOR", ["VS Code"]),
          ("OS", ["macOS · Linux"])]

SCRIPT = "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc"
ROUNDED = "/System/Library/Fonts/SFNSRounded.ttf"

OUT_W = 900
FRAMES, FRAME_MS = 160, 50               # 8 s loop
MARGIN = 40                              # off-screen run-up above and below

WHITE, PINK, GLOW, SHADOW = (255, 250, 253), (255, 196, 228), (255, 150, 210), (30, 20, 90)
PETAL = np.array([249, 163, 220.])       # the art's own petal colour
PETAL_BACK = np.array([244, 172, 230.])  # underside, a touch cooler
SKY = np.array([90, 82, 202.])

# petals painted into the source art (source px); the last three touch Sakura,
# so their fill has to leave her outline alone
PAINTED = [(242, 133), (1087, 267), (588, 272), (109, 293), (994, 327), (1236, 385),
           (1376, 493), (428, 520), (1073, 552), (992, 614), (679, 639), (577, 733),
           (677, 750), (74, 753), (919, 830), (1271, 889), (627, 296), (353, 336),
           (425, 391), (496, 459), (1050, 325), (610, 425), (974, 359), (1036, 492)]
TOUCHING = 3

# depth layers. run: vertical travel per loop in screen heights (above 1 the
# petal spends part of the loop off-screen, which is what lets it fall faster)
LAYERS = {
    "far":  dict(n=22, size=(10, 15), run=1.0, drift=0.22, sway=(4, 10), alpha=0.8, haze=0.3, blur=0),
    "mid":  dict(n=22, size=(20, 32), run=1.5, drift=0.30, sway=(8, 18), alpha=1.0, haze=0, blur=0),
    "near": dict(n=4, size=(36, 50), run=2.4, drift=0.40, sway=(14, 26), alpha=0.9, haze=0, blur=2.2),
}
SS = 4                                   # supersampling for petal edges


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


# ------------------------------------------------------------ the art ---

def petal_holes(img):
    """Per painted petal: the hole to fill (the petal and its glow) and, for
    the ones touching Sakura, the pixels of hers the fill must not cover."""
    H, W = img.shape[:2]
    pdist = np.sqrt(((img - PETAL) ** 2).sum(-1))
    seg = PETAL - SKY
    t = np.clip(np.einsum("...c,c->...", img - SKY, seg) / (seg * seg).sum(), 0, 1)
    off_line = np.sqrt(((img - SKY - t[..., None] * seg) ** 2).sum(-1))
    figure = ndi.binary_dilation(off_line > 48, disk(2))     # neither sky nor petal
    yy, xx = np.mgrid[0:H, 0:W]

    holes = []
    for k, (x, y) in enumerate(PAINTED):
        thr = max(42, pdist[y - 12:y + 13, x - 12:x + 13].min() + 22)
        lab, _ = ndi.label((pdist < thr) & ((xx - x) ** 2 + (yy - y) ** 2 < 42 ** 2))
        win = lab[y - 10:y + 11, x - 10:x + 11]
        core = np.isin(lab, np.unique(win[win > 0]))
        hole = ndi.binary_dilation(core, disk(int(round(5 + 0.14 * np.sqrt(core.sum())))))
        keep = np.zeros_like(core)
        if k >= len(PAINTED) - TOUCHING:
            keep = figure & ~ndi.binary_dilation(core, disk(2))
        holes.append((hole & ~keep, keep))
    return holes


def best_offset(img, taken, y0, x0, lh, lr):
    """Shift to the nearby patch whose rim best matches this one's, skipping
    patches that overlap any petal; shorter shifts win near-ties."""
    H, W = img.shape[:2]
    h, w = lh.shape
    target = img[y0:y0 + h, x0:x0 + w][lr]
    best = None
    for dy in range(-50, 51):
        for dx in range(-120, 121):
            sy, sx = y0 + dy, x0 + dx
            if abs(dx) < w // 2 and abs(dy) < h // 2 or sy < 0 or sx < 0 or sy + h > H or sx + w > W:
                continue
            if taken[sy:sy + h, sx:sx + w][lh | lr].any():
                continue
            e = ((img[sy:sy + h, sx:sx + w][lr] - target) ** 2).mean() + 0.02 * np.hypot(dx, dy)
            if best is None or e < best[0]:
                best = (e, dy, dx)
    return best[1], best[2]


def lift_petals(img):
    """Remove the painted petals, glow included. Each hole is filled from the
    nearby patch whose surroundings match best (so scanlines and windows line
    up), then the leftover mismatch on the rim is spread across the hole."""
    holes = petal_holes(img)
    taken = ndi.binary_dilation(np.any([h for h, _ in holes], axis=0), disk(3))
    out = img.copy()
    for hole, keep in holes:
        rim = ndi.binary_dilation(hole, disk(7)) & ~hole & ~keep
        ys, xs = np.nonzero(hole | rim)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lh, lr = hole[y0:y1, x0:x1], rim[y0:y1, x0:x1]
        dy, dx = best_offset(img, taken, y0, x0, lh, lr)
        src = img[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        rim_diff = np.where(lr[..., None], img[y0:y1, x0:x1] - src, 0)
        fix = np.zeros_like(src)
        for _ in range(400):                       # membrane: rim mismatch, relaxed inwards
            fix = np.where(lr[..., None], rim_diff, fix)
            fix = np.where(lh[..., None], ndi.uniform_filter(fix, (3, 3, 1)), fix)
        fix = ndi.gaussian_filter(fix, (2, 2, 0))
        out[y0:y1, x0:x1][lh] = (src + fix)[lh]
    return out


# --------------------------------------------------------------- text ---

def tracked(d, x, y, text, font, fill, track, right=False):
    if right:
        x -= d.textlength(text, font=font) + track * (len(text) - 1)
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track


def rounded(size, weight):
    f = ImageFont.truetype(ROUNDED, size)
    f.set_variation_by_axes([weight])
    return f


def text_layer(size, s=3):
    """Title top left, details down the right edge: white with a pink glow
    and a faint indigo shadow, as one RGBA layer. The type is drawn at s x
    and scaled down; hinted small sizes space letters unevenly."""
    W, H = size
    ink = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    title = ImageFont.truetype(SCRIPT, 104 * s, index=1)
    d.text((40 * s, 100 * s), TITLE, font=title, fill=WHITE, anchor="ls")
    tracked(d, 48 * s, 146 * s, TAGLINE, rounded(13 * s, 600), PINK, 2.6 * s)

    label, value = rounded(12 * s, 700), rounded(16 * s, 500)
    x, y = (W - 42) * s, 44 * s
    for name, vals in FIELDS:
        tracked(d, x, y, name, label, PINK, 2.4 * s, right=True)
        y += 18 * s
        for v in vals:
            d.text((x, y), v, font=value, fill=WHITE, anchor="ra")
            y += 22 * s
        y += 12 * s
    ink = ink.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")

    a = ink.getchannel("A")
    layer = Image.new("RGBA", size, SHADOW + (0,))
    layer.putalpha(a.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.35)))
    glow = Image.new("RGBA", size, GLOW + (0,))
    glow.putalpha(a.filter(ImageFilter.GaussianBlur(7)).point(lambda v: min(255, int(v * 1.2))))
    return Image.alpha_composite(Image.alpha_composite(layer, glow), ink)


# ------------------------------------------------------------- petals ---

def petal_outline(n=20):
    """Unit petal, as painted in the art: a lens pointed at both ends, one side
    a little fuller, with a tiny tail at the stem end. Long axis is x."""
    u = np.linspace(-1, 1, n)
    top = 0.55 * (1 - u ** 2) ** 0.9
    bot = -0.45 * (1 - u ** 2) ** 0.9
    return np.concatenate([np.stack([u, top], 1), [[1.12, 0.02]], np.stack([u[::-1], bot[::-1]], 1)])


UNIT = petal_outline()


def make_petals(seed=8):
    rng = np.random.default_rng(seed)
    petals = []
    for name, L in LAYERS.items():
        n = L["n"]
        if name == "near":                       # keep the big ones off her face
            xs = np.array([0.08, 0.22, 0.78, 0.92])[:n]
        else:
            xs = (np.arange(n) + rng.uniform(0, 1, n)) / n
            rng.shuffle(xs)
        for i in range(n):
            petals.append(dict(L, layer=name, x=xs[i], phase=(i * 0.618034 + rng.uniform(0, 0.1)) % 1,
                               length=rng.uniform(*L["size"]), sway_px=rng.uniform(*L["sway"]),
                               sway_f=rng.uniform(0.6, 1.4), sway_p=rng.uniform(0, 2 * np.pi),
                               spin=rng.choice([-1, 1]) * rng.uniform(0.3, 1.0), rot0=rng.uniform(0, 360),
                               flips=int(rng.integers(2, 5)), flip0=rng.uniform(0, 2 * np.pi)))
    return petals


def pose(p, t, W, H):
    """Where a petal is at loop time t (0..1). Each petal falls once per loop
    and wraps while off-screen, so the loop is seamless."""
    span = H + 2 * MARGIN
    path = p["run"] * span
    u = (t + p["phase"]) % 1.0
    y = -MARGIN - (path - span) / 2 + path * u
    wave = 2 * np.pi * p["sway_f"] * p["run"] * u + p["sway_p"]
    x = p["x"] * W + p["drift"] * (y - H / 2) + p["sway_px"] * np.sin(wave)
    theta = p["rot0"] + 360 * p["spin"] * p["run"] * u + 20 * np.cos(wave)    # rocks as it sways
    flip = p["flip0"] + 2 * np.pi * p["flips"] * p["run"] * u
    return x, y, theta, flip


def draw_petal(f, p, t):
    """Composite one petal into the float frame f, in place."""
    H, W = f.shape[:2]
    x, y, theta, flip = pose(p, t, W, H)
    pts = UNIT * (p["length"] / 2)
    pts[:, 1] *= np.sign(np.cos(flip)) * (0.25 + 0.75 * abs(np.cos(flip)))   # tumbling about its long axis
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    pts = pts @ np.array([[c, s], [-s, c]]) + (x, y)
    pad = 2 + int(np.ceil(2 * p["blur"]))
    x0, y0 = int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad
    x1, y1 = int(pts[:, 0].max()) + pad + 1, int(pts[:, 1].max()) + pad + 1
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return
    m = Image.new("L", ((x1 - x0) * SS, (y1 - y0) * SS), 0)
    ImageDraw.Draw(m).polygon([((px - x0) * SS, (py - y0) * SS) for px, py in pts], fill=255)
    m = m.resize((x1 - x0, y1 - y0), Image.BOX)
    if p["blur"]:
        m = m.filter(ImageFilter.GaussianBlur(p["blur"]))
    a = np.asarray(m, dtype=float) / 255
    col = (PETAL if np.cos(flip) >= 0 else PETAL_BACK) * (0.86 + 0.14 * abs(np.cos(flip)))
    col = col * (1 - p["haze"]) + SKY * p["haze"]

    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    a = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None]
    region = f[cy0:cy1, cx0:cx1]
    region[:] = region * (1 - a * p["alpha"]) + col * a * p["alpha"]


# -------------------------------------------------------------- frames ---

def render_frame(art, text, petals, i):
    """Far and mid petals pass behind the text, near ones in front of it."""
    t = i / FRAMES
    f = np.asarray(art, dtype=float).copy()
    for p in petals:
        if p["layer"] != "near":
            draw_petal(f, p, t)
    a = text[..., 3:] / 255
    f = f * (1 - a) + text[..., :3] * a
    for p in petals:
        if p["layer"] == "near":
            draw_petal(f, p, t)
    return Image.fromarray(np.clip(f + 0.5, 0, 255).astype(np.uint8))


def build_scene(art_path):
    src = np.asarray(Image.open(art_path).convert("RGB"), dtype=float)
    clean = Image.fromarray(np.clip(lift_petals(src) + 0.5, 0, 255).astype(np.uint8))
    out_h = round(clean.height * OUT_W / clean.width)
    art = clean.resize((OUT_W, out_h), Image.LANCZOS)
    return art, np.asarray(text_layer(art.size), dtype=float)


def _palette_from(pixels, n):
    """Median-cut palette of up to n colours for an (N, 3) pixel array, refined
    with k-means (noticeably less dither grain on the dress)."""
    side = int(np.ceil(np.sqrt(len(pixels))))
    img = Image.fromarray(np.resize(pixels, (side * side, 3)).reshape(side, side, 3).astype(np.uint8))
    q = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT, kmeans=3)
    return np.array(q.getpalette()[:n * 3]).reshape(-1, 3)


def save_gif(base, frames, path):
    """Quantise the still scene once (dithered, so the sky stays smooth), then
    remap only the pixels each frame changes. Static pixels keep identical
    indices frame to frame, so the encoder can skip them."""
    rgb = np.asarray(base)
    moved = np.concatenate([np.asarray(f)[np.any(np.asarray(f) != rgb, -1)] for f in frames[::8]])
    # fixed budgets, so the blue sky can't starve Sakura's pinks and reds
    px = rgb.reshape(-1, 3)
    warm = px[:, 0] >= px[:, 2]
    pal = np.concatenate([_palette_from(px[~warm], 115), _palette_from(px[warm], 100), _palette_from(moved, 40)])
    # the pads are never picked, which frees an index for transparency
    padded = np.concatenate([pal, np.repeat(pal[:1], 256 - len(pal), 0)])
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(padded.astype(np.uint8).flatten().tolist())
    q = base.quantize(palette=pal_img)
    pal_flat = q.getpalette()
    base_idx = np.asarray(q)
    tree = cKDTree(pal.astype(float))

    out = []
    for f in frames:
        a = np.asarray(f)
        changed = np.any(a != rgb, -1)
        idx = base_idx.copy()
        if changed.any():
            idx[changed] = tree.query(a[changed].astype(float))[1]
        im = Image.frombytes("P", base.size, idx.astype(np.uint8).tobytes())
        im.putpalette(pal_flat)
        out.append(im)
    # optimize=True makes unchanged pixels transparent in later frames
    out[0].save(path, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, optimize=True)


def build(art_path, out_path):
    art, text = build_scene(art_path)
    petals = make_petals()
    if not out_path.lower().endswith(".gif"):
        render_frame(art, text, petals, 0).save(out_path)
        print(f"wrote {out_path}")
        return
    base = render_frame(art, text, [], 0)
    frames = []
    for i in range(FRAMES):
        frames.append(render_frame(art, text, petals, i))
        print(f"  frame {i + 1}/{FRAMES}", end="\r", flush=True)
    print()
    save_gif(base, frames, out_path)
    print(f"wrote {out_path}  ({art.width}x{art.height}, {FRAMES} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "card.png")

"""
Bracelet charging animation — horizontal surface fill.

The fill boundary is a VERTICAL LINE sweeping left←right (or right→left).
Every bracelet pixel at X < boundary is illuminated — full strap height.
This is exactly a "battery charging bar" / "liquid light" effect.

Seamless loop: fill_ratio = (1 − cos(2π × progress)) / 2
  0 → dark, 0.5 → fully lit, 1 → dark again. Value and derivative
  both match at the boundary → no cut at the loop point.
"""
from PIL import Image, ImageFilter
import numpy as np
import math

# ── Load ──────────────────────────────────────────────────────────────────────
src = Image.open("Bracciale senza sfondo.png").convert("RGBA")
MAX_W = 360
if src.width > MAX_W:
    r = MAX_W / src.width
    src = src.resize((MAX_W, int(src.height * r)), Image.LANCZOS)

W, H = src.size
print(f"Size: {W}×{H}")

b_arr   = np.array(src, dtype=np.float32)
b_alpha = b_arr[:, :, 3] / 255.0
b_rgb   = b_arr[:, :, :3] / 255.0

# ── Per-pixel X position (0 = right edge, 1 = left edge) ─────────────────────
# Fill starts from the RIGHT side and expands leftward during charge.
xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)

# Normalised X: 0 at right side, 1 at left side
fill_pos = (1.0 - XX / (W - 1)).astype(np.float32)

# ── Halo bloom source (Gaussian falloff from bracelet edge) ───────────────────
# Build a simple distance-from-alpha-edge map for atmospheric bloom.
# We'll derive it from the alpha channel itself via erosion proxy.
alpha_u8   = (b_alpha * 255).astype(np.uint8)
alpha_pil  = Image.fromarray(alpha_u8, "L")
# Blur the alpha to get a soft halo weight outside/inside the strap
halo_pil   = alpha_pil.filter(ImageFilter.GaussianBlur(radius=20))
g_halo     = np.array(halo_pil, dtype=np.float32) / 255.0

# ── Animation ─────────────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 4.5
N_FRAMES   = int(FPS * DURATION_S)   # 108 frames
FRAME_MS   = int(1000 / FPS)

N_BLUR    = 9
BLUR_HALF = (1.0 / N_FRAMES) * 0.75  # smear over ±0.75 frame-steps

# Soft edge: fraction of full width. 0.12 = ~43 px of gradual transition.
EDGE_W = 0.12

# Colours
LIT_COLOR    = np.array([0.88, 0.95, 1.00], dtype=np.float32)  # soft cool-white on strap surface
CORONA_COLOR = np.array([0.65, 0.85, 1.00], dtype=np.float32)  # cyan-white at leading edge & bloom
LIT_STR    = 0.72   # surface glow strength — bracelet texture remains visible
EDGE_STR   = 0.55   # leading edge highlight strength
BLOOM_STR1 = 0.55   # tight bloom blend
BLOOM_STR2 = 0.35   # wide atmospheric bloom blend


def fill_mask(fill_ratio: float):
    """
    Returns (lit, edge) for a given fill_ratio in [0, 1].
    lit  : 0 (dark) … 1 (fully lit), full-height at each X column.
    edge : Gaussian peak at the moving fill boundary.
    """
    # Extend slightly so bracelet is fully dark at ratio=0 and fully lit at ratio=1.
    fill_ext = fill_ratio * (1.0 + EDGE_W) - EDGE_W * 0.5

    # Signed distance from boundary (positive = lit side, i.e. already filled)
    d = (fill_ext - fill_pos) / EDGE_W

    # Smooth linear ramp — no overshoot
    lit  = np.clip(d + 0.5, 0.0, 1.0).astype(np.float32)

    # Bright Gaussian at the boundary for the soft leading-edge highlight
    edge = np.exp(-d ** 2 / (2 * 0.6 ** 2)).astype(np.float32)

    return lit, edge


def make_frame(progress: float) -> Image.Image:
    """
    progress in [0, 1) — last frame is one step before the loop point.
    fill_ratio = (1 − cos(2π × p)) / 2 → ease-in / ease-out, seamless.
    """
    lit_acc  = np.zeros((H, W), dtype=np.float32)
    edge_acc = np.zeros((H, W), dtype=np.float32)

    for k in range(N_BLUR):
        t  = progress + (k / (N_BLUR - 1) - 0.5) * 2.0 * BLUR_HALF
        fr = (1.0 - math.cos(2.0 * math.pi * t)) / 2.0
        l, e = fill_mask(fr)
        lit_acc  += l
        edge_acc += e

    lit_acc  /= N_BLUR
    edge_acc /= N_BLUR

    # Mask to bracelet pixels only
    lit_surface  = lit_acc  * b_alpha
    edge_surface = np.clip(edge_acc * b_alpha * 1.4, 0.0, 1.0)
    halo         = np.clip(lit_surface * g_halo * 0.55 + edge_surface * g_halo * 0.45, 0.0, 1.0)

    # ── Screen-blend glow onto bracelet base ──────────────────────────────────
    out = b_rgb.copy()
    for ch in range(3):
        # Surface fill: soft diffused glow over the full strap height
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - lit_surface  * LIT_COLOR[ch]    * LIT_STR)
        # Leading edge: brighter cyan-white highlight at the boundary
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - edge_surface * CORONA_COLOR[ch] * EDGE_STR)
        # Soft halo
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - halo         * CORONA_COLOR[ch] * 0.30)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha
    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Two-pass bloom (tight crisp + wide atmospheric) ───────────────────────
    g_combined = np.clip(lit_surface * 0.75 + edge_surface * 0.95, 0.0, 1.0)
    ga = np.zeros((H, W, 4), dtype=np.float32)
    for ch in range(3):
        ga[:, :, ch] = g_combined * CORONA_COLOR[ch]
    ga[:, :, 3] = np.clip(g_combined * b_alpha, 0.0, 1.0)

    gp = Image.fromarray((ga * 255).astype(np.uint8), "RGBA")
    bt = np.array(gp.filter(ImageFilter.GaussianBlur(radius=14)), dtype=np.float32) / 255.0
    bw = np.array(gp.filter(ImageFilter.GaussianBlur(radius=34)), dtype=np.float32) / 255.0

    fa = np.array(frame, dtype=np.float32) / 255.0
    for ch in range(3):
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bt[:, :, ch] * BLOOM_STR1)
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bw[:, :, ch] * BLOOM_STR2)
    fa[:, :, 3] = b_alpha

    return Image.fromarray((np.clip(fa, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    p = i / N_FRAMES
    frames.append(make_frame(p))
    if i % 12 == 0:
        print(f"  frame {i+1}/{N_FRAMES}  ({p*100:.0f}%)")

frames[0].save(
    "bracelet-animated.png",
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=FRAME_MS,
    format="PNG",
)
print(f"\nDone — {N_FRAMES} frames × {FRAME_MS}ms — {W}×{H}px")

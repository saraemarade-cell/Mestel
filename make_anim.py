"""
Bracelet charging animation — curved fill following the bracelet shape.

KEY DESIGN:
  fill_pos = angular position around the ellipse (wraps full 360°).
  At every angle the ENTIRE strap cross-section (inner to outer edge)
  is lit together — pure b_alpha masking, no Gaussian stripe.
  The fill boundary is a curved line that follows the bracelet contour,
  not a straight vertical line.

Seamless loop: fill_ratio = (1 − cos(2π × progress)) / 2
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

# ── Ellipse geometry ──────────────────────────────────────────────────────────
CX, CY = W * 0.50, H * 0.52
RX, RY = W * 0.46, H * 0.26

xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)
nx = (XX - CX) / RX
ny = (YY - CY) / RY
pixel_angle = np.arctan2(ny, nx).astype(np.float32)   # −π … π

# ── Curved fill parameterisation ──────────────────────────────────────────────
# Starts at top-right (~50° above horizontal on right side) and sweeps
# clockwise around the full bracelet.
# fill_pos ∈ [0, 1]: 0 = start/seam at top-right, 1 = full circle.
START_ANGLE         = -math.pi * 0.28
pixel_angle_wrapped = ((pixel_angle - START_ANGLE) % (2.0 * math.pi)).astype(np.float32)
fill_pos            = (pixel_angle_wrapped / (2.0 * math.pi)).astype(np.float32)

# ── Atmospheric bloom source ──────────────────────────────────────────────────
# Soft blur of the alpha mask → glow diffuses a few pixels beyond the strap.
alpha_pil = Image.fromarray((b_alpha * 255).astype(np.uint8), "L")
halo_soft = np.array(alpha_pil.filter(ImageFilter.GaussianBlur(radius=22)),
                     dtype=np.float32) / 255.0

# ── Animation settings ────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 4.5
N_FRAMES   = int(FPS * DURATION_S)   # 108 frames
FRAME_MS   = int(1000 / FPS)

N_BLUR    = 9
BLUR_HALF = (1.0 / N_FRAMES) * 0.75

# Soft edge width as fraction of full arc [0,1]
EDGE_W = 0.10

# Colours
LIT_COLOR    = np.array([0.88, 0.95, 1.00], dtype=np.float32)   # soft cool-white
CORONA_COLOR = np.array([0.60, 0.82, 1.00], dtype=np.float32)   # cyan-white bloom
LIT_STR    = 0.75
EDGE_STR   = 0.55
BLOOM_S1   = 0.55
BLOOM_S2   = 0.35


def fill_mask(fill_ratio: float):
    """
    lit  : 0…1, full-height across the strap at each angular position.
    edge : Gaussian peak at the curved fill boundary.
    """
    fill_ext = fill_ratio * (1.0 + EDGE_W) - EDGE_W * 0.5
    d   = (fill_ext - fill_pos) / EDGE_W
    lit = np.clip(d + 0.5, 0.0, 1.0).astype(np.float32)
    edge = np.exp(-d ** 2 / (2 * 0.55 ** 2)).astype(np.float32)
    return lit, edge


def make_frame(progress: float) -> Image.Image:
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

    # ── Full strap coverage: use ONLY b_alpha, no Gaussian stripe ────────────
    lit_surface  = lit_acc  * b_alpha
    edge_surface = np.clip(edge_acc * b_alpha * 1.4, 0.0, 1.0)

    # Atmospheric halo: lit area bled outward through blurred alpha
    halo = np.clip(
        lit_surface  * halo_soft * 0.60 +
        edge_surface * halo_soft * 0.50,
        0.0, 1.0
    )

    # ── Screen-blend glow onto the bracelet surface ───────────────────────────
    out = b_rgb.copy()
    for ch in range(3):
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - lit_surface  * LIT_COLOR[ch]    * LIT_STR)
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - edge_surface * CORONA_COLOR[ch] * EDGE_STR)
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - halo         * CORONA_COLOR[ch] * 0.30)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha
    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Two-pass atmospheric bloom ────────────────────────────────────────────
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
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bt[:, :, ch] * BLOOM_S1)
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bw[:, :, ch] * BLOOM_S2)
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

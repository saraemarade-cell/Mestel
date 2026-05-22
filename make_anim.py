"""
Bracelet charging animation — front face fill, left → right, then reverse.

Geometry fix: in image coordinates y increases downward, so the FRONT FACE
(lower portion of image) has pixel_angle ∈ [0, π].  The correct front-arc
parameterisation is fill_pos = (π − angle) / π → 0 at left, 1 at right.

Loop guarantee: fill_ratio = (1 − cos(2π × progress)) / 2
  progress=0   → fill_ratio=0  (dark, derivative=0)
  progress=0.5 → fill_ratio=1  (fully lit)
  progress=1   → fill_ratio=0  (dark again, derivative=0)
Both value AND derivative match at the loop boundary → perfectly seamless.
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
RX, RY = W * 0.45, H * 0.25

xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)
nx = (XX - CX) / RX
ny = (YY - CY) / RY

# In image coords y increases downward.
# Front face (lower image) → ny > 0 → pixel_angle ∈ (0, π)
# Back / top rim           → ny < 0 → pixel_angle ∈ (−π, 0)
pixel_angle = np.arctan2(ny, nx).astype(np.float32)

# ── Full-circle parameterisation ─────────────────────────────────────────────
# Fill wraps the ENTIRE bracelet starting from the right side (angle=0),
# going clockwise: right → bottom-front → left → top-back → right.
# fill_pos ∈ [0,1]: 0=right, 0.25=bottom, 0.5=left, 0.75=top, 1.0=full circle
pixel_angle_wrapped = (pixel_angle % (2.0 * math.pi)).astype(np.float32)
fill_pos = (pixel_angle_wrapped / (2.0 * math.pi)).astype(np.float32)

# Distance from ellipse centre-line (for the halo bloom that extends beyond strap)
ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)
dist_ellipse = np.sqrt((XX - ex)**2 + (YY - ey)**2).astype(np.float32)
HALO_W = 36.0
g_halo  = np.exp(-dist_ellipse**2 / (2 * HALO_W**2)).astype(np.float32)

# ── Animation settings ────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 4.5                       # one charge + discharge cycle
N_FRAMES   = int(FPS * DURATION_S)     # 108 frames
FRAME_MS   = int(1000 / FPS)           # ~41 ms

# Temporal motion blur: average N_BLUR fill positions per frame
N_BLUR    = 9
BLUR_HALF = (1.0 / N_FRAMES) * 0.75   # smear ±0.75 frame-steps

# Soft edge width as fraction of total arc
EDGE_W = 0.16

# Colours
WHITE  = np.array([1.00, 1.00, 1.00], dtype=np.float32)
CORONA = np.array([0.82, 0.95, 1.00], dtype=np.float32)  # very light cool-white


# ── Core fill function ────────────────────────────────────────────────────────
def fill_mask(fill_ratio: float):
    """
    Returns lit [0…1] and edge_peak [0…1] for a given fill_ratio ∈ [0,1].
    Extends fill by EDGE_W/2 beyond [0,1] so the bracelet is fully dark at 0
    and fully lit at 1 with no half-lit border artefact.
    """
    fill_ext = fill_ratio * (1.0 + EDGE_W) - EDGE_W * 0.5

    # Signed distance to fill boundary: positive = lit side
    d = (fill_ext - fill_pos) / EDGE_W

    # Linear ramp clamped to [0,1] — smooth but free of cosine overshoot
    lit  = np.clip(d + 0.5, 0.0, 1.0).astype(np.float32)

    # Gaussian peak right at the boundary for the bright leading edge
    edge = np.exp(-d**2 / (2 * 0.65**2)).astype(np.float32)

    return lit, edge


# ── Frame renderer ────────────────────────────────────────────────────────────
def make_frame(progress: float) -> Image.Image:
    """
    progress ∈ [0, 1).
    fill_ratio = (1 − cos(2π × p)) / 2  →  ease-in/ease-out, seamless loop.
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

    # Mask to bracelet silhouette
    lit_surface  = lit_acc  * b_alpha
    edge_surface = np.clip(edge_acc * b_alpha * 1.6, 0.0, 1.0)

    # Wide halo — soft glow that bleeds slightly beyond the strap edges
    halo = np.clip(
        lit_surface  * g_halo * 0.60 +
        edge_surface * g_halo * 0.55,
        0.0, 1.0
    )

    # ── Screen-blend layers onto the bracelet surface ─────────────────────────
    out = b_rgb.copy()
    for ch in range(3):
        # 1. Lit region → saturates to bright white
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - lit_surface  * WHITE[ch]  * 1.20)
        # 2. Leading-edge corona → cool-white accent
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - edge_surface * CORONA[ch] * 0.55)
        # 3. Halo beyond strap
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - halo         * CORONA[ch] * 0.35)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha
    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Two-pass bloom (tight crisp + wide atmospheric) ───────────────────────
    g_combined   = np.clip(lit_surface * 0.80 + edge_surface * 1.00, 0.0, 1.0)
    glow_arr     = np.zeros((H, W, 4), dtype=np.float32)
    for ch in range(3):
        glow_arr[:, :, ch] = g_combined * CORONA[ch]
    glow_arr[:, :, 3] = np.clip(g_combined * b_alpha, 0.0, 1.0)

    gp          = Image.fromarray((glow_arr * 255).astype(np.uint8), "RGBA")
    bloom_tight = gp.filter(ImageFilter.GaussianBlur(radius=14))
    bloom_wide  = gp.filter(ImageFilter.GaussianBlur(radius=36))

    fa = np.array(frame,       dtype=np.float32) / 255.0
    bt = np.array(bloom_tight, dtype=np.float32) / 255.0
    bw = np.array(bloom_wide,  dtype=np.float32) / 255.0

    for ch in range(3):
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bt[:, :, ch] * 0.78)
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bw[:, :, ch] * 0.52)
    fa[:, :, 3] = b_alpha

    return Image.fromarray((np.clip(fa, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    p = i / N_FRAMES            # i/N → last frame is one step before loop point
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

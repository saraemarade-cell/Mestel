"""
Seamless bracelet animation — liquid-light quality.

Techniques for maximum smoothness:
  - Temporal motion blur: each frame averages N_BLUR sub-positions so the
    band edge is never a hard line
  - Exponential tail decay: no cutoff angle, glow fades continuously to zero
  - Gaussian angular head: smooth bell-curve profile, no clipping artifacts
  - Wide soft halo for bloom
  - Perfectly seamless loop: progress = i/N (last frame is one step before t=1)
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
CX, CY = W * 0.50, H * 0.50
RX, RY = W * 0.44, H * 0.22

xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)
nx = (XX - CX) / RX
ny = (YY - CY) / RY
pixel_angle = np.arctan2(ny, nx).astype(np.float32)   # -π … π

ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)
dist_ellipse = np.sqrt((XX - ex) ** 2 + (YY - ey) ** 2).astype(np.float32)
HALO_W = 36.0
g_halo = np.exp(-dist_ellipse ** 2 / (2 * HALO_W ** 2)).astype(np.float32)

# ── Animation settings ─────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 3.5
N_FRAMES   = int(FPS * DURATION_S)   # 84 frames
FRAME_MS   = int(1000 / FPS)          # ~42 ms

# Motion blur: average N_BLUR angular sub-steps per rendered frame.
# blur_spread = fraction of one frame's angular step to smear over.
N_BLUR     = 7
STEP_ANGLE = (2 * math.pi) / N_FRAMES   # angular distance per frame
blur_spread = STEP_ANGLE * 1.4           # smear over ~1.4 frame-steps

# Glow profile parameters
# Tail: cosine ease over a fixed arc, then ZERO — no accumulation
TAIL_ANGLE = math.pi * 0.55   # trailing arc = ~100°, rest of bracelet stays dark
HEAD_SIGMA = 0.38             # Gaussian σ for the ignition front (radians)
HEAD_BOOST = 2.2              # peak brightness at the leading edge

# ── Colour palette (matches reference: white core → cyan-blue → deep blue) ───
# Strap surface when lit: near-pure white with the faintest cool blue tint
WHITE_SURF  = np.array([0.93, 0.97, 1.00], dtype=np.float32)
# Ignition front: pure white-hot
WHITE_HOT   = np.array([1.00, 1.00, 1.00], dtype=np.float32)
# Inner corona just outside/around the strap: electric cyan-blue
CYAN_CORONA = np.array([0.12, 0.80, 1.00], dtype=np.float32)   # #1FCCFF
# Outer bloom atmosphere: deeper cool blue
BLUE_BLOOM  = np.array([0.06, 0.52, 1.00], dtype=np.float32)   # #1085FF


def glow_fields(band_center: float):
    """
    Compute (tail_field, head_field) for a single band position.
    Both are masked to the bracelet alpha so they fill the exact strap shape.
    """
    TWO_PI = 2 * math.pi

    # Angular distance behind the head for every pixel (0 = just passed)
    d_behind = ((band_center - pixel_angle) % TWO_PI).astype(np.float32)

    # ── Tail: cosine ease from 1→0 over TAIL_ANGLE, then hard zero ──────────
    # This gives a single finite arc of light — no accumulation anywhere.
    # cos ease: smooth derivative at both ends, reaches exactly 0 at TAIL_ANGLE.
    tail = np.where(
        d_behind < TAIL_ANGLE,
        (0.5 * (1.0 + np.cos(math.pi * d_behind / TAIL_ANGLE))).astype(np.float32),
        0.0
    ).astype(np.float32)

    # ── Head: Gaussian angular profile — smooth bell curve ───────────────────
    ang_dist = np.abs(
        ((pixel_angle - band_center) + math.pi) % TWO_PI - math.pi
    ).astype(np.float32)
    head = np.exp(-ang_dist ** 2 / (2 * HEAD_SIGMA ** 2)).astype(np.float32)

    # Mask both to bracelet silhouette
    return tail * b_alpha, head * b_alpha


def make_frame(progress: float) -> Image.Image:
    """
    Render one frame with temporal motion blur.
    progress in [0, 1): at 0 and 1 band_center differs by exactly 2π → seamless.
    """
    band_center_base = progress * 2 * math.pi - math.pi / 2

    # Accumulate glow over N_BLUR sub-steps
    tail_acc = np.zeros((H, W), dtype=np.float32)
    head_acc = np.zeros((H, W), dtype=np.float32)

    for k in range(N_BLUR):
        t = k / (N_BLUR - 1) if N_BLUR > 1 else 0.5
        offset = (t - 0.5) * blur_spread
        t_f, h_f = glow_fields(band_center_base + offset)
        tail_acc += t_f
        head_acc += h_f

    tail_acc /= N_BLUR
    head_acc /= N_BLUR

    head_bright = np.clip(head_acc * HEAD_BOOST, 0.0, 1.0)
    glow_total  = np.clip(tail_acc + head_bright, 0.0, 1.0)

    # Wide halo for bloom source (uses the averaged band center)
    halo_field = np.clip(
        tail_acc * g_halo * 0.55 + head_acc * g_halo * b_alpha * 1.3,
        0.0, 1.0
    ).astype(np.float32)

    # ── Layer 1: strap surface → bright white (screen blend) ─────────────────
    out = b_rgb.copy()
    for ch in range(3):
        # Tail behind band: strap surface goes white-blue
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - tail_acc * WHITE_SURF[ch] * 1.1)
        # Head front: strap surface goes pure white-hot
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - head_bright * WHITE_HOT[ch])

    # ── Layer 2: cyan-blue corona at strap edges (halo field) ────────────────
    for ch in range(3):
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - halo_field * CYAN_CORONA[ch] * 0.70)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha
    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Layer 3: deep-blue atmospheric bloom (blurred, screen-blended back) ──
    # Two bloom passes: tight cyan-blue + wide deep-blue atmosphere
    g_combined   = np.clip(glow_total + halo_field * 0.6, 0.0, 1.0)
    glow_img_arr = np.zeros((H, W, 4), dtype=np.float32)
    for ch in range(3):
        # Blend between CYAN_CORONA (bright areas) and BLUE_BLOOM (faint areas)
        glow_img_arr[:, :, ch] = (
            glow_total * CYAN_CORONA[ch] * 0.6 +
            halo_field * BLUE_BLOOM[ch]  * 0.8
        )
    glow_img_arr[:, :, 3] = np.clip(g_combined * b_alpha, 0.0, 1.0)

    # Two radii: tight (crisp cyan edge) + wide (blue atmosphere)
    bloom_tight = Image.fromarray((glow_img_arr * 255).astype(np.uint8), "RGBA") \
                       .filter(ImageFilter.GaussianBlur(radius=10))
    bloom_wide  = Image.fromarray((glow_img_arr * 255).astype(np.uint8), "RGBA") \
                       .filter(ImageFilter.GaussianBlur(radius=28))

    frame_arr       = np.array(frame,       dtype=np.float32) / 255.0
    bloom_tight_arr = np.array(bloom_tight, dtype=np.float32) / 255.0
    bloom_wide_arr  = np.array(bloom_wide,  dtype=np.float32) / 255.0

    for ch in range(3):
        frame_arr[:, :, ch] = 1.0 - (1.0 - frame_arr[:, :, ch]) * (1.0 - bloom_tight_arr[:, :, ch] * 0.75)
        frame_arr[:, :, ch] = 1.0 - (1.0 - frame_arr[:, :, ch]) * (1.0 - bloom_wide_arr[:, :, ch]  * 0.55)
    frame_arr[:, :, 3] = b_alpha

    return Image.fromarray((np.clip(frame_arr, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")


# ── Render ─────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    p = i / N_FRAMES   # i/N not i/(N-1) — last frame is one step before loop point
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

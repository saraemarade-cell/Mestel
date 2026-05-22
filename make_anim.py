"""
Bracelet charging animation.

Behavior:
  - The front visible face of the bracelet fills with light from left → right.
  - Once fully lit, the light shrinks back right → left (discharge).
  - Repeat infinitely — seamless, no cuts.

Technique:
  - Pixels are parameterised by their angular position along the FRONT ARC
    of the ellipse (lower half, angle ∈ [−π, 0]), mapped to [0, 1] left→right.
  - fill_ratio = (1 − cos(2π × progress)) / 2  →  0 at start/end, 1 at midpoint.
    The cosine gives natural ease-in/ease-out AND guarantees seamless looping
    because the value and its derivative are both identical at progress=0 and 1.
  - A soft linear ramp (width = EDGE_W arc-fraction) creates the smooth
    transition edge — no hard lines.
  - White luminous core + light cyan-white bloom at the leading edge.
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
RX, RY = W * 0.44, H * 0.24

xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)
nx = (XX - CX) / RX
ny = (YY - CY) / RY
pixel_angle = np.arctan2(ny, nx).astype(np.float32)   # −π … π

# Distance from ellipse centre-line (for halo bloom)
ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)
dist_ellipse = np.sqrt((XX - ex)**2 + (YY - ey)**2).astype(np.float32)
HALO_W = 32.0
g_halo = np.exp(-dist_ellipse**2 / (2 * HALO_W**2)).astype(np.float32)

# ── Front-arc parameterisation ────────────────────────────────────────────────
# Front visible face = lower half of ellipse = angles in [−π, 0].
# Map to fill_pos ∈ [0, 1] where 0 = leftmost point, 1 = rightmost.
is_front = (pixel_angle <= 0.0).astype(np.float32)          # 1 on front, 0 on back
fill_pos  = (pixel_angle + math.pi) / math.pi               # [0,1] on front, [1,2] on back
# Back pixels get a fill_pos > 1 so they are never lit by the fill mask.

# ── Animation settings ────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 4.0                          # one charge + discharge cycle
N_FRAMES   = int(FPS * DURATION_S)        # 96 frames
FRAME_MS   = int(1000 / FPS)              # ~42 ms

# Motion-blur sub-steps: average N_BLUR fill positions centred on each frame
# so the moving edge is always smooth even at speed.
N_BLUR     = 7
STEP       = 1.0 / N_FRAMES              # one frame's worth of progress
BLUR_HALF  = STEP * 0.7                  # half-width of blur window

# Soft edge: ramp width as a fraction of the full arc [0, 1]
EDGE_W = 0.11

# Colours
WHITE_CORE  = np.array([1.00, 1.00, 1.00], dtype=np.float32)
EDGE_GLOW_C = np.array([0.88, 0.97, 1.00], dtype=np.float32)  # very slight cool-white at edge


def fill_mask(fill_ratio: float):
    """
    Returns (lit, edge_glow) arrays for a given fill_ratio ∈ [0, 1].
    fill_ratio 0 → fully dark, fill_ratio 1 → fully lit.

    lit       : 0…1, 1 = fully illuminated bracelet pixel
    edge_glow : peaks at the moving boundary, 0 elsewhere
    """
    # Extend the fill slightly beyond [0,1] so bracelet is fully dark at ratio=0
    # and fully lit at ratio=1, with no half-lit edge hanging off either extreme.
    fill_ext = fill_ratio * (1.0 + EDGE_W) - EDGE_W / 2.0

    # Signed distance to fill boundary (positive = lit side)
    d = (fill_ext - fill_pos) / EDGE_W                  # ∈ (−∞, +∞)

    lit  = np.clip(d + 0.5, 0.0, 1.0).astype(np.float32)
    lit *= is_front                                       # back of bracelet stays dark

    # Gaussian peak at the boundary (d = 0) for the bright leading edge
    edge = np.exp(-d**2 / (2 * 0.55**2)).astype(np.float32) * is_front

    return lit, edge


def make_frame(progress: float) -> Image.Image:
    """
    progress ∈ [0, 1).
    fill_ratio = (1 − cos(2π × progress)) / 2 gives:
      0 at progress 0  (dark)
      1 at progress 0.5 (fully lit)
      0 at progress 1  (dark again — seamless loop)
    """
    # Accumulate over N_BLUR sub-steps for temporal motion blur
    lit_acc  = np.zeros((H, W), dtype=np.float32)
    edge_acc = np.zeros((H, W), dtype=np.float32)

    for k in range(N_BLUR):
        t = progress + (k / (N_BLUR - 1) - 0.5) * 2 * BLUR_HALF
        fr = (1.0 - math.cos(2 * math.pi * t)) / 2.0
        l, e = fill_mask(fr)
        lit_acc  += l
        edge_acc += e

    lit_acc  /= N_BLUR
    edge_acc /= N_BLUR

    # Mask to bracelet silhouette
    lit_surface  = lit_acc  * b_alpha
    edge_surface = np.clip(edge_acc * b_alpha * 1.4, 0.0, 1.0)

    # Halo extends slightly beyond the strap edges
    halo = np.clip(lit_surface * g_halo * 0.55 + edge_surface * g_halo * 0.50, 0.0, 1.0)

    # ── Compose onto bracelet base ────────────────────────────────────────────
    out = b_rgb.copy()
    for ch in range(3):
        # Lit surface → bright white
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - lit_surface  * WHITE_CORE[ch]  * 1.10)
        # Leading edge → cool-white corona
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - edge_surface * EDGE_GLOW_C[ch] * 0.45)
        # Halo
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - halo         * EDGE_GLOW_C[ch] * 0.30)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha
    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Bloom: two-pass Gaussian, screen-blended back ─────────────────────────
    g_combined   = np.clip(lit_surface * 0.75 + edge_surface * 0.90, 0.0, 1.0)
    glow_arr     = np.zeros((H, W, 4), dtype=np.float32)
    for ch in range(3):
        glow_arr[:, :, ch] = g_combined * EDGE_GLOW_C[ch]
    glow_arr[:, :, 3] = np.clip(g_combined * b_alpha, 0.0, 1.0)

    glow_pil    = Image.fromarray((glow_arr * 255).astype(np.uint8), "RGBA")
    bloom_tight = glow_pil.filter(ImageFilter.GaussianBlur(radius=12))
    bloom_wide  = glow_pil.filter(ImageFilter.GaussianBlur(radius=30))

    fa  = np.array(frame,       dtype=np.float32) / 255.0
    bt  = np.array(bloom_tight, dtype=np.float32) / 255.0
    bw  = np.array(bloom_wide,  dtype=np.float32) / 255.0

    for ch in range(3):
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bt[:, :, ch] * 0.72)
        fa[:, :, ch] = 1.0 - (1.0 - fa[:, :, ch]) * (1.0 - bw[:, :, ch] * 0.48)
    fa[:, :, 3] = b_alpha

    return Image.fromarray((np.clip(fa, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    p = i / N_FRAMES        # i/N (not i/(N−1)) → last frame is one step before loop point
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

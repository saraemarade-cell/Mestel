"""
Bracelet light band — continuous smooth approach.
For each pixel we compute:
  1. Its angular position projected onto the ellipse  → angle
  2. Its distance from the ellipse surface            → radial falloff (Gaussian)
  3. Whether it falls inside the moving band window   → angular intensity
No discrete dots; the glow is a single continuous mathematical field.
"""
from PIL import Image
import numpy as np
import math

# ── Load & resize ────────────────────────────────────────────────────────────
src = Image.open("Bracciale senza sfondo.png").convert("RGBA")
MAX_W = 600
if src.width > MAX_W:
    r = MAX_W / src.width
    src = src.resize((MAX_W, int(src.height * r)), Image.LANCZOS)

W, H = src.size
print(f"Size: {W}×{H}")

bracelet_arr   = np.array(src, dtype=np.float32)
bracelet_alpha = bracelet_arr[:, :, 3] / 255.0          # 0-1 mask

# ── Ellipse parameters ────────────────────────────────────────────────────────
CX, CY = W * 0.50, H * 0.50
RX, RY = W * 0.40, H * 0.20

# ── Pre-compute per-pixel ellipse geometry (done ONCE) ───────────────────────
xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)

nx = (XX - CX) / RX                     # normalised x  (-1…+1 on ellipse)
ny = (YY - CY) / RY                     # normalised y

pixel_angle = np.arctan2(ny, nx)        # angle of projection onto ellipse

# Closest point on ellipse at that angle
ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)

# Euclidean distance from ellipse surface (pixels)
dist_from_ellipse = np.sqrt((XX - ex) ** 2 + (YY - ey) ** 2)

# ── Animation parameters ──────────────────────────────────────────────────────
N_FRAMES   = 60
FRAME_MS   = 67          # 60 × 67 ms ≈ 4 s

TAIL_SPAN  = 1.6         # radians of tail behind head (~92°)
HEAD_SPAN  = 0.10        # small forward glow

BAND_W_PX  = 14.0        # half-width of light band in pixels (Gaussian sigma)
CORE_W_PX  = 6.0         # tighter bright core

# Radial Gaussians (precomputed)
radial_outer = np.exp(-dist_from_ellipse ** 2 / (2 * BAND_W_PX ** 2))
radial_core  = np.exp(-dist_from_ellipse ** 2 / (2 * CORE_W_PX ** 2))

def make_frame(progress):
    band_center = progress * 2 * math.pi - math.pi / 2

    # Angular distance from band center, wrapped to (-π, π)
    d_angle = pixel_angle - band_center
    d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi

    # --- Angular intensity: 1 at head, smooth fade toward tail, 0 outside ---
    in_band = (d_angle >= -TAIL_SPAN) & (d_angle <= HEAD_SPAN)

    # Position within band: 0 = tail end, 1 = head
    band_pos = np.clip(
        (d_angle + TAIL_SPAN) / (TAIL_SPAN + HEAD_SPAN), 0, 1
    )
    angular = np.where(in_band, band_pos ** 1.4, 0.0)   # smooth ramp

    # --- Combine angular × radial → continuous field ---
    field_outer = angular * radial_outer   # soft halo
    field_core  = angular * radial_core   # bright core

    # Clip to bracelet shape
    field_outer *= bracelet_alpha
    field_core  *= bracelet_alpha

    # --- Compose over bracelet image ---
    out = bracelet_arr.copy() / 255.0

    # Teal: R≈0  G≈0.9  B≈0.78  (normalised from #00E5C8)
    for ch, (o_mult, c_mult) in enumerate(
        [(0.30, 0.55), (0.85, 1.00), (0.75, 0.92), (0.50, 1.00)]
    ):
        contrib = field_outer * o_mult + field_core * c_mult
        if ch < 3:
            # Screen blend for colour channels
            out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - contrib)
        # Alpha stays bracelet_alpha

    out[:, :, 3] = bracelet_alpha
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    frames.append(make_frame(i / N_FRAMES))
    if i % 15 == 0:
        print(f"  frame {i}/{N_FRAMES}")

frames[0].save(
    "bracelet-animated.png",
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=FRAME_MS,
    format="PNG",
)
print(f"Saved — {N_FRAMES} frames × {FRAME_MS} ms — {W}×{H} px")

"""
Cinematic bracelet animation:
- Phase 1: bracelet fades in matte black
- Phase 2: energy band ignites and sweeps around leaving activated glow
- Phase 3: fully illuminated, gentle pulse
- Bloom via multi-layer Gaussian
"""
from PIL import Image, ImageFilter
import numpy as np
import math

# ── Load ─────────────────────────────────────────────────────────────────────
src = Image.open("Bracciale senza sfondo.png").convert("RGBA")
MAX_W = 360
if src.width > MAX_W:
    r = MAX_W / src.width
    src = src.resize((MAX_W, int(src.height * r)), Image.LANCZOS)

W, H = src.size
print(f"Size: {W}×{H}")

b_arr   = np.array(src, dtype=np.float32)
b_alpha = b_arr[:, :, 3] / 255.0
b_rgb   = b_arr[:, :, :3] / 255.0   # original colour (matte dark)

# ── Ellipse ───────────────────────────────────────────────────────────────────
CX, CY = W * 0.50, H * 0.50
RX, RY = W * 0.40, H * 0.20

# ── Pre-compute pixel geometry (once) ────────────────────────────────────────
xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)

nx = (XX - CX) / RX
ny = (YY - CY) / RY
pixel_angle = np.arctan2(ny, nx)          # angle of closest ellipse point

ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)
dist_ellipse = np.sqrt((XX - ex)**2 + (YY - ey)**2)  # px distance from curve

# Radial masks (Gaussians pre-baked)
BAND_W  = 12.0
CORE_W  =  5.0
HALO_W  = 28.0
g_band  = np.exp(-dist_ellipse**2 / (2 * BAND_W**2))
g_core  = np.exp(-dist_ellipse**2 / (2 * CORE_W**2))
g_halo  = np.exp(-dist_ellipse**2 / (2 * HALO_W**2))

# ── Animation settings ────────────────────────────────────────────────────────
FPS         = 20
DURATION_S  = 4.0
N_FRAMES    = int(FPS * DURATION_S)
FRAME_MS    = int(1000 / FPS)

FADE_IN_END   = 0.12   # bracelet fades in (progress 0-0.12)
IGNITE_START  = 0.14   # band appears
SWEEP_END     = 0.80   # band completes one full revolution
PULSE_START   = 0.82   # final glow pulse


def angular_behind(a, band_center):
    """Radians behind the band head, wrapped 0-2π."""
    d = (band_center - a) % (2 * math.pi)
    return d


def make_frame(progress):
    # ── 1. Bracelet opacity (fade-in) ─────────────────────────────────────────
    if progress < FADE_IN_END:
        bracelet_opacity = progress / FADE_IN_END
    else:
        bracelet_opacity = 1.0

    # ── 2. Band position & activation map ────────────────────────────────────
    if progress < IGNITE_START:
        sweep_t = 0.0
    elif progress <= SWEEP_END:
        sweep_t = (progress - IGNITE_START) / (SWEEP_END - IGNITE_START)
    else:
        sweep_t = 1.0

    band_center = sweep_t * 2 * math.pi - math.pi / 2

    # Angular distance behind band for every pixel
    d_behind = (band_center - pixel_angle) % (2 * math.pi)   # 0=just passed → 2π=ahead

    # --- Activation map: how much glow the surface emits (0-1) ---
    # Pixels that have been passed by the band glow, intensity decays with distance
    activated = np.where(
        sweep_t > 0,
        np.clip(1.0 - (d_behind / (2 * math.pi)) ** 0.45, 0, 1),
        0.0
    )
    # Pixels AHEAD of the band (d_behind close to 2π) haven't been activated
    activated = np.where(d_behind > (2 * math.pi - 0.25), 0.0, activated)

    # During pulse phase, whole bracelet glows uniformly
    if progress > PULSE_START:
        pulse = (progress - PULSE_START) / (1.0 - PULSE_START)
        pulse_intensity = 0.35 + 0.15 * math.sin(pulse * math.pi * 2)
        activated = np.maximum(activated, pulse_intensity)

    # --- Moving band at the head ---
    # HEAD: the bright ignition front
    # Tight angular window around band_center
    HEAD_HALF = 0.18    # radians
    ang_dist_from_head = np.abs(((pixel_angle - band_center) + math.pi) % (2*math.pi) - math.pi)
    head_angular = np.clip(1.0 - ang_dist_from_head / HEAD_HALF, 0, 1) ** 1.2

    # ── 3. Compose layers ─────────────────────────────────────────────────────
    # Layer A: dimmed matte bracelet
    # Layer B: activated cyan glow surface
    # Layer C: bright ignition head
    # Layer D: bloom (blurred glow)

    # A — base bracelet (slightly darkened when activated around it)
    base = b_rgb.copy()

    # B — surface activation glow (teal tint over the bracelet)
    act_field    = activated * g_band         # spread to band-width
    act_field   *= b_alpha

    # C — ignition head (very bright)
    if sweep_t > 0:
        head_field = head_angular * g_core * b_alpha * 1.8
    else:
        head_field = np.zeros((H, W), dtype=np.float32)

    # Combined glow intensity
    glow_total = np.clip(act_field + head_field, 0, 1)

    # Add wide halo for bloom
    halo_field = np.clip(
        activated * g_halo * 0.5 + head_angular * g_halo * b_alpha * 1.2, 0, 1
    )

    # Screen-blend teal colour onto bracelet
    out = base.copy()
    teal = np.array([0.0, 0.898, 0.784], dtype=np.float32)   # #00E5C8
    white_core = np.array([0.85, 1.0, 0.97], dtype=np.float32)

    for ch in range(3):
        # Surface activation layer
        contrib = glow_total * teal[ch] + halo_field * teal[ch] * 0.4
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - contrib)
        # White-hot core at ignition front
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - head_field * white_core[ch] * 0.6)

    # ── 4. PIL frame: apply bracelet_opacity, bloom ───────────────────────────
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0, 1)
    rgba[:, :, 3]  = b_alpha * bracelet_opacity

    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # Bloom: blur the glow, add on top with screen
    # Build a pure-glow RGBA image for blur
    glow_img_arr = np.zeros((H, W, 4), dtype=np.float32)
    g_combined = np.clip(glow_total + halo_field * 0.6, 0, 1)
    for ch in range(3):
        glow_img_arr[:, :, ch] = g_combined * teal[ch]
    glow_img_arr[:, :, 3] = np.clip(g_combined * b_alpha, 0, 1)

    glow_pil = Image.fromarray((glow_img_arr * 255).astype(np.uint8), "RGBA")
    bloom    = glow_pil.filter(ImageFilter.GaussianBlur(radius=18))

    # Composite: frame + bloom (screen)
    frame_arr  = np.array(frame,  dtype=np.float32) / 255.0
    bloom_arr  = np.array(bloom,  dtype=np.float32) / 255.0

    for ch in range(3):
        frame_arr[:, :, ch] = 1.0 - (1.0 - frame_arr[:, :, ch]) * (1.0 - bloom_arr[:, :, ch] * 0.7)

    frame_arr[:, :, 3] = b_alpha * bracelet_opacity
    return Image.fromarray((np.clip(frame_arr, 0, 1) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    p = i / N_FRAMES
    frames.append(make_frame(p))
    if i % 15 == 0:
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

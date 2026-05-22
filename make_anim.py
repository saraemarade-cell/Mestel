"""
Seamless bracelet animation — perfectly continuous infinite loop.

The band rotates at constant speed. Frame 0 and frame N share the same
angular state (band_center differs by exactly 2π), so playback loops
without any cut, jump, or pause.

No phase transitions. No fade-in. Constant motion.
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
b_rgb   = b_arr[:, :, :3] / 255.0

# ── Ellipse geometry ──────────────────────────────────────────────────────────
CX, CY = W * 0.50, H * 0.50
RX, RY = W * 0.40, H * 0.20

xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)

nx = (XX - CX) / RX
ny = (YY - CY) / RY
pixel_angle = np.arctan2(ny, nx).astype(np.float32)  # -π to π, per-pixel

# Halo bloom mask (distance from ellipse centre-line)
ex = CX + RX * np.cos(pixel_angle)
ey = CY + RY * np.sin(pixel_angle)
dist_ellipse = np.sqrt((XX - ex) ** 2 + (YY - ey) ** 2)
HALO_W  = 26.0
g_halo  = np.exp(-dist_ellipse ** 2 / (2 * HALO_W ** 2)).astype(np.float32)

# ── Animation settings ────────────────────────────────────────────────────────
FPS        = 24
DURATION_S = 3.0
N_FRAMES   = int(FPS * DURATION_S)   # 72 frames
FRAME_MS   = int(1000 / FPS)          # ~42 ms

# Glow parameters
TAIL_ANGLE = math.pi * 1.35   # trailing glow arc (243°) — leaves a dark zone ahead
HEAD_HALF  = 0.30              # half-width of bright ignition front (radians)

teal       = np.array([0.0, 0.898, 0.784], dtype=np.float32)  # #00E5C8
white_core = np.array([0.85, 1.0,  0.97],  dtype=np.float32)


def make_frame(progress: float) -> Image.Image:
    """
    progress in [0, 1). At progress=0 and progress=1 the band_center
    differs by exactly 2π, guaranteeing frame-perfect seamless looping.
    """
    # ── Band position (constant angular velocity) ─────────────────────────────
    band_center = progress * 2 * math.pi - math.pi / 2   # starts at top of ellipse

    # Angular distance BEHIND the head for every pixel (0 = just passed, 2π = just ahead)
    d_behind = ((band_center - pixel_angle) % (2 * math.pi)).astype(np.float32)

    # ── Activation tail ───────────────────────────────────────────────────────
    # Pixels within TAIL_ANGLE behind the head glow; intensity falls off smoothly.
    activated = np.where(
        d_behind < TAIL_ANGLE,
        np.clip(np.maximum(0.0, 1.0 - d_behind / TAIL_ANGLE) ** 0.55, 0.0, 1.0),
        0.0
    ).astype(np.float32)

    # ── Ignition front ────────────────────────────────────────────────────────
    ang_dist_from_head = np.abs(
        ((pixel_angle - band_center) + math.pi) % (2 * math.pi) - math.pi
    ).astype(np.float32)
    head_angular = np.clip(1.0 - ang_dist_from_head / HEAD_HALF, 0.0, 1.0).astype(np.float32) ** 0.8

    # ── Glow fields masked to exact bracelet strap silhouette ─────────────────
    act_field  = activated    * b_alpha          # trailing glow — full strap width
    head_field = head_angular * b_alpha * 1.8    # ignition front — full strap width
    glow_total = np.clip(act_field + head_field, 0.0, 1.0)

    # Wide halo extends slightly beyond strap edges (cinematic bloom source)
    halo_field = np.clip(
        activated    * g_halo * 0.5 +
        head_angular * g_halo * b_alpha * 1.2,
        0.0, 1.0
    ).astype(np.float32)

    # ── Screen-blend teal + white-hot core onto base bracelet ─────────────────
    out = b_rgb.copy()
    for ch in range(3):
        contrib = glow_total * teal[ch] + halo_field * teal[ch] * 0.4
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - contrib)
        out[:, :, ch] = 1.0 - (1.0 - out[:, :, ch]) * (1.0 - head_field * white_core[ch] * 0.6)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(out, 0.0, 1.0)
    rgba[:, :, 3]  = b_alpha

    frame = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")

    # ── Bloom: blur glow layer, screen-blend back ─────────────────────────────
    g_combined   = np.clip(glow_total + halo_field * 0.6, 0.0, 1.0)
    glow_img_arr = np.zeros((H, W, 4), dtype=np.float32)
    for ch in range(3):
        glow_img_arr[:, :, ch] = g_combined * teal[ch]
    glow_img_arr[:, :, 3] = np.clip(g_combined * b_alpha, 0.0, 1.0)

    bloom     = Image.fromarray((glow_img_arr * 255).astype(np.uint8), "RGBA") \
                     .filter(ImageFilter.GaussianBlur(radius=16))

    frame_arr = np.array(frame, dtype=np.float32) / 255.0
    bloom_arr = np.array(bloom, dtype=np.float32) / 255.0

    for ch in range(3):
        frame_arr[:, :, ch] = 1.0 - (1.0 - frame_arr[:, :, ch]) * (1.0 - bloom_arr[:, :, ch] * 0.7)
    frame_arr[:, :, 3] = b_alpha

    return Image.fromarray((np.clip(frame_arr, 0.0, 1.0) * 255).astype(np.uint8), "RGBA")


# ── Render ────────────────────────────────────────────────────────────────────
frames = []
for i in range(N_FRAMES):
    # progress = i/N (NOT i/(N-1)) so the last frame is one step BEFORE
    # the loop point — the next frame would be frame 0 again.
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

from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import math

src = Image.open("bracciale.png").convert("RGBA")

# Ridimensiona per web (max 600px larghezza)
MAX_W = 600
if src.width > MAX_W:
    ratio = MAX_W / src.width
    src = src.resize((MAX_W, int(src.height * ratio)), Image.LANCZOS)

W, H = src.size
print(f"Working size: {W}x{H}")

bracelet_arr   = np.array(src, dtype=np.float32)
bracelet_alpha = bracelet_arr[:, :, 3] / 255.0

CX = W * 0.50
CY = H * 0.50
RX = W * 0.40
RY = H * 0.20

N_FRAMES  = 48
FRAME_MS  = 83          # 48 × 83ms ≈ 4s
TAIL      = 12
GLOW_R    = int(W * 0.09)

def make_frame(progress):
    angle = progress * 2 * math.pi - math.pi / 2

    glow = np.zeros((H, W, 4), dtype=np.float32)

    for t in range(TAIL, -1, -1):
        a    = angle - t * 0.25
        px   = CX + RX * math.cos(a)
        py   = CY + RY * math.sin(a)
        frac = 1.0 - t / TAIL
        alpha = frac ** 1.4
        size  = max(4, int(GLOW_R * (0.45 + 0.55 * frac)))

        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)

        halo = np.clip(1.0 - dist / (size * 2.2), 0, 1) ** 2.0 * alpha * 0.45
        core = np.clip(1.0 - dist / size,          0, 1) ** 1.5 * alpha

        glow[:, :, 0] += core * 0.55 + halo * 0.18
        glow[:, :, 1] += core * 1.00 + halo * 0.80
        glow[:, :, 2] += core * 0.92 + halo * 0.75
        glow[:, :, 3] += core * 1.00 + halo * 0.45

    glow = np.clip(glow, 0, 1)
    glow[:, :, 3] *= bracelet_alpha

    out = bracelet_arr.copy() / 255.0
    out[:, :, 0] = 1.0 - (1.0 - out[:, :, 0]) * (1.0 - glow[:, :, 0])
    out[:, :, 1] = 1.0 - (1.0 - out[:, :, 1]) * (1.0 - glow[:, :, 1])
    out[:, :, 2] = 1.0 - (1.0 - out[:, :, 2]) * (1.0 - glow[:, :, 2])
    out[:, :, 3] = bracelet_alpha

    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")


frames = []
for i in range(N_FRAMES):
    frames.append(make_frame(i / N_FRAMES))
    if i % 12 == 0:
        print(f"  frame {i}/{N_FRAMES}")

frames[0].save(
    "bracelet-animated.png",
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=FRAME_MS,
    format="PNG",
)
print(f"Done: {N_FRAMES} frames × {FRAME_MS}ms — {W}x{H}px")


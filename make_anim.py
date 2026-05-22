from PIL import Image
import numpy as np
import math

src = Image.open("Bracciale senza sfondo.png").convert("RGBA")

MAX_W = 600
if src.width > MAX_W:
    ratio = MAX_W / src.width
    src = src.resize((MAX_W, int(src.height * ratio)), Image.LANCZOS)

W, H = src.size
print(f"Working size: {W}x{H}")

bracelet_arr   = np.array(src, dtype=np.float32)
bracelet_alpha = bracelet_arr[:, :, 3] / 255.0

# Ellipse path along the bracelet
CX = W * 0.50
CY = H * 0.50
RX = W * 0.40
RY = H * 0.20

N_FRAMES  = 60
FRAME_MS  = 67          # 60 × 67ms ≈ 4s loop
GLOW_R    = int(W * 0.08)

# Pre-compute meshgrid once
xs = np.arange(W, dtype=np.float32)
ys = np.arange(H, dtype=np.float32)
XX, YY = np.meshgrid(xs, ys)

def make_frame(progress):
    angle = progress * 2 * math.pi - math.pi / 2

    glow = np.zeros((H, W, 4), dtype=np.float32)

    # Continuous scia: dense trail covering ~1/3 of the ellipse
    TRAIL_STEPS = 28
    TRAIL_SPAN  = 1.8   # radians (roughly 100° of the ellipse lit up)

    for t in range(TRAIL_STEPS, -1, -1):
        frac  = t / TRAIL_STEPS          # 1 = head, 0 = tail end
        a     = angle - (1.0 - frac) * TRAIL_SPAN
        px    = CX + RX * math.cos(a)
        py    = CY + RY * math.sin(a)

        # Intensity curve: bright at head, smooth fade toward tail
        intensity = frac ** 1.2
        size      = max(3, int(GLOW_R * (0.4 + 0.6 * frac)))

        dist = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)

        outer = np.clip(1.0 - dist / (size * 2.5), 0, 1) ** 2.0 * intensity * 0.4
        core  = np.clip(1.0 - dist / size,          0, 1) ** 1.5 * intensity

        glow[:, :, 0] += core * 0.55 + outer * 0.15
        glow[:, :, 1] += core * 1.00 + outer * 0.80
        glow[:, :, 2] += core * 0.92 + outer * 0.75
        glow[:, :, 3] += core * 1.00 + outer * 0.45

    glow = np.clip(glow, 0, 1)
    # Mask strictly to bracelet alpha
    glow[:, :, 3] *= bracelet_alpha

    out = bracelet_arr.copy() / 255.0
    # Screen blend
    out[:, :, 0] = 1.0 - (1.0 - out[:, :, 0]) * (1.0 - glow[:, :, 0])
    out[:, :, 1] = 1.0 - (1.0 - out[:, :, 1]) * (1.0 - glow[:, :, 1])
    out[:, :, 2] = 1.0 - (1.0 - out[:, :, 2]) * (1.0 - glow[:, :, 2])
    out[:, :, 3] = bracelet_alpha

    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")


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
print(f"Done: {N_FRAMES} frames × {FRAME_MS}ms — {W}x{H}px")

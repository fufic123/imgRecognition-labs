import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import random

OUT = os.path.join(os.path.dirname(__file__), "output")


def save(name, img, cmap=None):
    path = os.path.join(OUT, name)
    if cmap == "gray" or (len(img.shape) == 2):
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, img)
    print(f"  saved: {name}")
    return path


def savefig(name, fig):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")
    return path


def show_grid(images, titles, filename, cmap_list=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title, cmap in zip(axes, images, titles, cmap_list or [None] * n):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    savefig(filename, fig)


# ── Generate a sample dataset (5 images) ──────────────────────────────────────

def make_sample_images():
    imgs = []
    rng = np.random.default_rng(42)

    for i in range(5):
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        # background gradient
        for y in range(256):
            canvas[y, :] = [int(y * 0.8), int((255 - y) * 0.6), 120 + i * 20]
        # shapes
        color1 = tuple(int(c) for c in rng.integers(50, 255, 3))
        color2 = tuple(int(c) for c in rng.integers(50, 255, 3))
        cx, cy = rng.integers(60, 196, 2).tolist()
        r = int(rng.integers(30, 70))
        cv2.circle(canvas, (cx, cy), r, color1, -1)
        x1, y1 = rng.integers(20, 100, 2).tolist()
        x2, y2 = rng.integers(140, 230, 2).tolist()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color2, -1)
        # add noise to make blur/filter steps visible
        noise = rng.integers(0, 30, canvas.shape, dtype=np.uint8)
        canvas = cv2.add(canvas, noise)
        imgs.append(canvas)

    return imgs


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("Generating sample dataset...")
    originals = make_sample_images()
    img = originals[0]  # work with first image for step-by-step output

    # ── Step 1: Resize ────────────────────────────────────────────────────────
    print("\n[Step 1] Resize to 128x128")
    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    save("step1_original_256x256.png", img)
    save("step1_resized_128x128.png", resized)

    resized_all = [cv2.resize(i, (128, 128)) for i in originals]
    save("step1_resized_sample2.png", resized_all[1])
    save("step1_resized_sample3.png", resized_all[2])

    show_grid(
        [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)],
        ["Original (256×256)", "Resized (128×128)"],
        "step1_resize_comparison.png"
    )

    # ── Step 2: Color model conversion ───────────────────────────────────────
    print("\n[Step 2] Color space conversions")

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    save("step2_grayscale.png", gray)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    save("step2_hsv_raw.png", hsv)
    # split channels for visualization
    h, s, v = cv2.split(hsv)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original RGB")
    axes[1].imshow(h, cmap="hsv");    axes[1].set_title("HSV — Hue")
    axes[2].imshow(s, cmap="gray");   axes[2].set_title("HSV — Saturation")
    axes[3].imshow(v, cmap="gray");   axes[3].set_title("HSV — Value")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    savefig("step2_hsv_channels.png", fig)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    save("step2_lab_raw.png", lab)
    l, a, b = cv2.split(lab)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original RGB")
    axes[1].imshow(l, cmap="gray");  axes[1].set_title("LAB — L (Lightness)")
    axes[2].imshow(a, cmap="RdYlGn_r"); axes[2].set_title("LAB — A (Green→Red)")
    axes[3].imshow(b, cmap="RdYlBu_r"); axes[3].set_title("LAB — B (Blue→Yellow)")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    savefig("step2_lab_channels.png", fig)

    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), gray, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)],
        ["Original RGB", "Grayscale", "HSV (converted back)", "LAB (converted back)"],
        "step2_color_models_overview.png",
        cmap_list=[None, "gray", None, None]
    )

    # ── Step 3: Augmentations ─────────────────────────────────────────────────
    print("\n[Step 3] Augmentations")

    # 3a — Random rotation
    angle = random.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (128, 128), borderMode=cv2.BORDER_REFLECT)
    save("step3a_rotation.png", rotated)
    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)],
        ["Before", f"Rotated ({angle:.1f}°)"],
        "step3a_rotation_comparison.png"
    )

    # 3b — Median Blur (3x3 and 5x5)
    median_3 = cv2.medianBlur(resized, 3)
    median_5 = cv2.medianBlur(resized, 5)
    save("step3b_median_blur_3x3.png", median_3)
    save("step3b_median_blur_5x5.png", median_5)
    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(median_3, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(median_5, cv2.COLOR_BGR2RGB)],
        ["Original", "Median Blur 3×3", "Median Blur 5×5"],
        "step3b_median_blur_comparison.png"
    )

    # 3c — Random brightness
    factor = random.uniform(0.5, 1.8)
    bright = cv2.convertScaleAbs(resized, alpha=factor, beta=0)
    save("step3c_brightness.png", bright)
    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)],
        ["Before", f"Brightness ×{factor:.2f}"],
        "step3c_brightness_comparison.png"
    )

    # 3d — Horizontal and vertical flips
    flip_h = cv2.flip(resized, 1)
    flip_v = cv2.flip(resized, 0)
    save("step3d_flip_horizontal.png", flip_h)
    save("step3d_flip_vertical.png", flip_v)
    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(flip_h, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(flip_v, cv2.COLOR_BGR2RGB)],
        ["Original", "Horizontal Flip", "Vertical Flip"],
        "step3d_flips_comparison.png"
    )

    # 3e — Bilateral filter (d=5 and d=9)
    bilateral_5 = cv2.bilateralFilter(resized, d=5,  sigmaColor=75, sigmaSpace=75)
    bilateral_9 = cv2.bilateralFilter(resized, d=9,  sigmaColor=75, sigmaSpace=75)
    save("step3e_bilateral_d5.png",  bilateral_5)
    save("step3e_bilateral_d9.png",  bilateral_9)
    show_grid(
        [cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(bilateral_5, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(bilateral_9, cv2.COLOR_BGR2RGB)],
        ["Original", "Bilateral d=5", "Bilateral d=9"],
        "step3e_bilateral_comparison.png"
    )

    # ── Full pipeline overview ────────────────────────────────────────────────
    print("\n[Overview] Full pipeline grid")
    all_imgs = [
        cv2.cvtColor(resized,    cv2.COLOR_BGR2RGB),
        gray,
        cv2.cvtColor(rotated,    cv2.COLOR_BGR2RGB),
        cv2.cvtColor(median_3,   cv2.COLOR_BGR2RGB),
        cv2.cvtColor(median_5,   cv2.COLOR_BGR2RGB),
        cv2.cvtColor(bright,     cv2.COLOR_BGR2RGB),
        cv2.cvtColor(flip_h,     cv2.COLOR_BGR2RGB),
        cv2.cvtColor(flip_v,     cv2.COLOR_BGR2RGB),
        cv2.cvtColor(bilateral_5,cv2.COLOR_BGR2RGB),
        cv2.cvtColor(bilateral_9,cv2.COLOR_BGR2RGB),
    ]
    all_titles = [
        "Resized", "Grayscale",
        f"Rotation {angle:.1f}°",
        "Median 3×3", "Median 5×5",
        f"Brightness ×{factor:.2f}",
        "Flip H", "Flip V",
        "Bilateral d=5", "Bilateral d=9",
    ]
    all_cmaps = [None, "gray"] + [None] * 8

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, img_i, title, cmap in zip(axes.flat, all_imgs, all_titles, all_cmaps):
        ax.imshow(img_i, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Lab 2 — Full Pipeline Overview", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig("overview_full_pipeline.png", fig)

    print("\nDone. All outputs saved to:", OUT)


if __name__ == "__main__":
    main()

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT     = os.path.join(os.path.dirname(__file__), "output")
IMG_SRC = os.path.join(os.path.dirname(__file__), "input_handwritten.png")


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f"  saved: {name}")
    return path


def savefig(name, fig):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def show_comparison(images, titles, filename, cmaps=None, figsize=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 5))
    if n == 1:
        axes = [axes]
    cmaps = cmaps or [None] * n
    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    savefig(filename, fig)


def main():
    # ── Load image ────────────────────────────────────────────────────────────
    print("[Input] Loading handwritten photo")
    img_bgr = cv2.imread(IMG_SRC)
    assert img_bgr is not None, f"Cannot load image: {IMG_SRC}"
    save("input_handwritten.png", img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    save("input_grayscale.png", gray)

    # ── Step 1: Color Spectrogram Analysis ───────────────────────────────────
    print("\n[Step 1] Color spectrogram analysis")

    # Grayscale histogram + OTSU threshold
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist, color="steelblue", linewidth=1.2)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Grayscale Intensity Histogram (OTSU threshold)")
    ax.axvline(otsu_thresh, color="crimson", linestyle="--", linewidth=1.5,
               label=f"OTSU threshold = {int(otsu_thresh)}")
    ax.fill_betweenx([0, hist.max()], 0, otsu_thresh,
                     alpha=0.08, color="black", label="Text region (dark)")
    ax.fill_betweenx([0, hist.max()], otsu_thresh, 255,
                     alpha=0.08, color="gold",  label="Background (light)")
    ax.legend()
    fig.tight_layout()
    savefig("step1_grayscale_histogram.png", fig)

    # RGB channel histograms
    channels = cv2.split(img_bgr)
    channel_info = [("Blue", "royalblue"), ("Green", "seagreen"), ("Red", "firebrick")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ch, (name, col) in zip(axes, channels, channel_info):
        h = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        ax.plot(h, color=col)
        ax.set_xlim(0, 255)
        ax.set_title(f"{name} channel histogram")
        ax.axvline(otsu_thresh, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"T={int(otsu_thresh)}")
        ax.legend(fontsize=8)
    fig.suptitle("RGB Channel Histograms — Threshold Analysis", fontsize=11)
    fig.tight_layout()
    savefig("step1_rgb_histograms.png", fig)

    # Pixel intensity scatter (column vs intensity)
    flat  = gray.flatten()
    y_pos = np.repeat(np.arange(gray.shape[0]), gray.shape[1])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(flat[::80], y_pos[::80], c=flat[::80],
               cmap="gray_r", s=1, alpha=0.4)
    ax.axvline(otsu_thresh, color="crimson", linestyle="--", linewidth=1.5,
               label=f"OTSU T={int(otsu_thresh)}")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Row (y)")
    ax.set_title("Pixel Intensity Scatter — Text vs Background")
    ax.legend()
    fig.tight_layout()
    savefig("step1_intensity_scatter.png", fig)

    print(f"  OTSU threshold: {int(otsu_thresh)}")

    # ── Step 2: Background Removal ───────────────────────────────────────────
    print("\n[Step 2] Background removal")

    # Adaptive threshold works better on grid-paper photos
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    # Clean up speckles
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    save("step2_binary_mask.png", binary_clean)

    # White background version
    mask3        = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
    bg_white     = np.full_like(img_bgr, 255)
    text_on_white = np.where(mask3 > 0, img_bgr, bg_white)
    save("step2_background_removed.png", text_on_white)

    # Transparent (BGRA) version
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = binary_clean
    save("step2_background_removed_alpha.png", bgra)

    show_comparison(
        [img_rgb, binary_clean,
         cv2.cvtColor(text_on_white, cv2.COLOR_BGR2RGB)],
        ["Original", f"Binary Mask (adaptive, T≈{int(otsu_thresh)})",
         "Background Removed"],
        "step2_background_removal_comparison.png",
        cmaps=[None, "gray", None]
    )

    # ── Step 3: Edge Detection ────────────────────────────────────────────────
    print("\n[Step 3] Edge detection")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny
    canny = cv2.Canny(blurred, 30, 100)
    save("step3_edges_canny.png", canny)

    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel  = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
    save("step3_edges_sobel.png", sobel)

    # Harris corners
    harris     = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    harris     = cv2.dilate(harris, None)
    harris_vis = img_bgr.copy()
    harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]
    save("step3_harris_corners.png", harris_vis)

    # SIFT keypoints
    sift     = cv2.SIFT_create()
    kp, _    = sift.detectAndCompute(gray, None)
    sift_vis = cv2.drawKeypoints(img_bgr, kp, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save("step3_sift_keypoints.png", sift_vis)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (im, t, cm) in zip(axes, [
        (img_rgb,                                      "Original",       None),
        (canny,                                        "Canny Edges",    "gray"),
        (sobel,                                        "Sobel Edges",    "gray"),
        (cv2.cvtColor(harris_vis, cv2.COLOR_BGR2RGB),  "Harris Corners", None),
    ]):
        ax.imshow(im, cmap=cm); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.tight_layout()
    savefig("step3_edge_detection_comparison.png", fig)
    print(f"  SIFT keypoints detected: {len(kp)}")

    # ── Step 4: Segmentation ──────────────────────────────────────────────────
    print("\n[Step 4] Image segmentation")

    edge_dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=2)
    seg_mask = cv2.bitwise_or(binary_clean, edge_dilated)
    seg_mask = cv2.morphologyEx(seg_mask,
                                cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                iterations=2)
    save("step4_segmentation_mask.png", seg_mask)

    seg_overlay = img_bgr.copy()
    seg_overlay[seg_mask > 0] = (
        seg_overlay[seg_mask > 0] * 0.4 + np.array([0, 180, 0]) * 0.6
    ).astype(np.uint8)
    save("step4_segmentation_overlay.png", seg_overlay)

    show_comparison(
        [canny, binary_clean, seg_mask,
         cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)],
        ["Canny Edges", "Binary Mask", "Segmentation Mask", "Overlay"],
        "step4_segmentation_comparison.png",
        cmaps=["gray", "gray", "gray", None]
    )

    # ── Step 5: Mask Application ──────────────────────────────────────────────
    print("\n[Step 5] Segmentation mask application")

    mask3_seg = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

    # Variation A: text pixels only (black background)
    text_only = np.where(mask3_seg > 0, img_bgr, np.zeros_like(img_bgr))
    save("step5_text_only.png", text_only)

    # Variation B: text sharp, background dimmed + cyan contours
    dimmed      = (img_bgr * 0.25).astype(np.uint8)
    highlighted = np.where(mask3_seg > 0, img_bgr, dimmed)
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlighted, contours, -1, (0, 220, 200), 1)
    save("step5_text_highlighted.png", highlighted)

    show_comparison(
        [img_rgb,
         cv2.cvtColor(text_only,   cv2.COLOR_BGR2RGB),
         cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)],
        ["Original", "Text Pixels Only", "Text Highlighted + Dimmed BG"],
        "step5_mask_application_comparison.png"
    )

    # ── Full pipeline overview ────────────────────────────────────────────────
    print("\n[Overview] Full pipeline grid")
    overview = [
        (img_rgb,                                          "Input",           None),
        (gray,                                             "Grayscale",       "gray"),
        (binary_clean,                                     "Binary Mask",     "gray"),
        (cv2.cvtColor(text_on_white, cv2.COLOR_BGR2RGB),   "BG Removed",      None),
        (canny,                                            "Canny Edges",     "gray"),
        (sobel,                                            "Sobel Edges",     "gray"),
        (cv2.cvtColor(harris_vis, cv2.COLOR_BGR2RGB),      "Harris Corners",  None),
        (cv2.cvtColor(sift_vis,   cv2.COLOR_BGR2RGB),      "SIFT Keypoints",  None),
        (seg_mask,                                         "Seg Mask",        "gray"),
        (cv2.cvtColor(seg_overlay,   cv2.COLOR_BGR2RGB),   "Seg Overlay",     None),
        (cv2.cvtColor(text_only,     cv2.COLOR_BGR2RGB),   "Text Only",       None),
        (cv2.cvtColor(highlighted,   cv2.COLOR_BGR2RGB),   "Highlighted",     None),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for ax, (im, t, cm) in zip(axes.flat, overview):
        ax.imshow(im, cmap=cm); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle("Lab 3 — Feature Extraction Full Pipeline", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig("overview_full_pipeline.png", fig)

    print("\nDone. All outputs saved to:", OUT)


if __name__ == "__main__":
    main()

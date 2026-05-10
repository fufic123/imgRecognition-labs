import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = os.path.join(os.path.dirname(__file__), "output")

# Generate a sample image if none provided
img_path = os.path.join(os.path.dirname(__file__), "image.jpg")
if not os.path.exists(img_path):
    sample = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(sample, (200, 150), 80, (255, 100, 50), -1)
    cv2.rectangle(sample, (50, 50), (150, 250), (50, 200, 255), -1)
    cv2.putText(sample, "Sample", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(img_path, sample)
    print("No image.jpg found — generated a sample image.")

img = cv2.imread(img_path)

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Histogram equalization
equalized = cv2.equalizeHist(gray)

# 3. Edge detection
edges = cv2.Canny(gray, 100, 200)

# 4. Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Convert BGR → RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

titles = ["Original", "Grayscale", "Equalized", "Edges", "Blurred"]
images = [img_rgb, gray, equalized, edges, blurred_rgb]
cmaps = [None, "gray", "gray", "gray", None]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, title, image, cmap in zip(axes, titles, images, cmaps):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(output_dir, "result.png")
plt.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")

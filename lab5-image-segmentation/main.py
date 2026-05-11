import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

OUT         = os.path.join(os.path.dirname(__file__), "output")
IMG_SIZE    = 128
NUM_CLASSES = 3
BATCH_SIZE  = 16
AUTOTUNE    = tf.data.AUTOTUNE

# Pet=green, Background=blue, Border=yellow
PALETTE     = np.array([[67, 160, 71], [33, 150, 243], [255, 193, 7]], dtype=np.uint8)
CLASS_NAMES = ["Pet", "Background", "Border"]


def savefig(name, fig):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in enumerate(PALETTE):
        rgb[mask == c] = color
    return rgb


# ── Step 1: Load dataset ──────────────────────────────────────────────────────
print("[Step 1] Loading Oxford-IIIT Pet dataset")

dataset, info = tfds.load("oxford_iiit_pet", with_info=True)
print(f"  Train: {info.splits['train'].num_examples}")
print(f"  Test:  {info.splits['test'].num_examples}")


def preprocess(sample):
    image = tf.image.resize(tf.cast(sample["image"], tf.float32), [IMG_SIZE, IMG_SIZE]) / 255.0
    mask  = tf.image.resize(
        sample["segmentation_mask"], [IMG_SIZE, IMG_SIZE], method="nearest")
    mask  = tf.cast(mask, tf.int32) - 1   # [1,2,3] → [0,1,2]
    mask  = tf.squeeze(mask, axis=-1)      # (H,W,1) → (H,W)
    return image, mask


train_ds = (dataset["train"]
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE))
test_ds  = (dataset["test"]
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# Dataset samples grid (2 per class × 3 classes + others)
sample_batch = list(train_ds.unbatch().take(8))
s_imgs  = np.stack([x[0].numpy() for x in sample_batch])
s_masks = np.stack([x[1].numpy() for x in sample_batch])

fig, axes = plt.subplots(2, 8, figsize=(22, 6))
for i in range(8):
    axes[0][i].imshow(s_imgs[i]);            axes[0][i].axis("off")
    axes[1][i].imshow(mask_to_rgb(s_masks[i])); axes[1][i].axis("off")
axes[0][0].set_ylabel("Image",  fontsize=10, rotation=0, labelpad=40, va="center")
axes[1][0].set_ylabel("Mask",   fontsize=10, rotation=0, labelpad=40, va="center")
handles = [plt.Rectangle((0,0),1,1, color=c/255, label=n)
           for c, n in zip(PALETTE, CLASS_NAMES)]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Step 1 — Oxford-IIIT Pet Dataset (3680 train / 3669 test)", fontsize=11)
fig.tight_layout()
savefig("step1_dataset_samples.png", fig)


# ── Step 2: Normalize ─────────────────────────────────────────────────────────
print("\n[Step 2] Normalization")

raw_batch = next(iter(dataset["train"].map(lambda s: {
    "image": tf.image.resize(tf.cast(s["image"], tf.float32), [IMG_SIZE, IMG_SIZE]),
    "segmentation_mask": tf.image.resize(s["segmentation_mask"], [IMG_SIZE, IMG_SIZE], method="nearest"),
}).batch(3)))
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i in range(3):
    raw = tf.image.resize(tf.cast(raw_batch["image"][i], tf.float32), [IMG_SIZE, IMG_SIZE])
    nrm = raw / 255.0
    axes[0][i].imshow(raw.numpy().astype(np.uint8))
    axes[0][i].set_title(f"Raw  [{int(raw.numpy().min())}–{int(raw.numpy().max())}]", fontsize=9)
    axes[0][i].axis("off")
    axes[1][i].imshow(nrm.numpy())
    axes[1][i].set_title(f"Normalized [0.0–1.0]", fontsize=9)
    axes[1][i].axis("off")
axes[0][0].set_ylabel("Before", fontsize=10)
axes[1][0].set_ylabel("After",  fontsize=10)
fig.suptitle("Step 2 — Pixel Normalization (÷255)", fontsize=11)
fig.tight_layout()
savefig("step2_normalization.png", fig)


# ── Step 3: Visualize images + masks + overlay ────────────────────────────────
print("\n[Step 3] Visualizing images alongside segmentation masks")

viz_samples = list(train_ds.unbatch().take(4))
v_imgs  = np.stack([x[0].numpy() for x in viz_samples])
v_masks = np.stack([x[1].numpy() for x in viz_samples])

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    axes[0][i].imshow(v_imgs[i]);                axes[0][i].axis("off")
    axes[1][i].imshow(mask_to_rgb(v_masks[i]));  axes[1][i].axis("off")
    overlay = np.clip(v_imgs[i] * 0.55 + mask_to_rgb(v_masks[i]) / 255.0 * 0.45, 0, 1)
    axes[2][i].imshow(overlay);                  axes[2][i].axis("off")

row_labels = ["Input Image", "Ground Truth Mask", "Overlay"]
for row, label in enumerate(row_labels):
    axes[row][0].set_ylabel(label, fontsize=10, rotation=0, labelpad=95, va="center")

handles = [plt.Rectangle((0,0),1,1, color=c/255, label=n)
           for c, n in zip(PALETTE, CLASS_NAMES)]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Step 3 — Input Images vs Segmentation Masks", fontsize=12)
fig.tight_layout()
savefig("step3_visualization.png", fig)


# ── Step 4: Build U-Net Xception ──────────────────────────────────────────────
print("\n[Step 4] Building U-Net with Xception backbone")


def build_unet_xception(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    h = w = img_size
    inputs = tf.keras.Input(shape=(h, w, 3), name="input_image")

    # Scale [0,1] → [-1,1] as Xception expects
    x_pre = inputs * 2.0 - 1.0

    # Pretrained Xception encoder (frozen)
    base = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=(h, w, 3)
    )
    base.trainable = False

    # For 128x128 input Xception produces: 16x16 and 8x8 at clean scale boundaries
    # (first two blocks use valid padding → odd sizes 63,61 unsuitable for concat)
    skip_names = ["block4_sepconv2_bn", "block13_sepconv2_bn"]
    print(f"  Skip layers (16x16, 8x8): {skip_names}")

    encoder = tf.keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(n).output for n in skip_names] + [base.output],
        name="xception_encoder"
    )

    feats      = encoder(x_pre)
    skip_16    = feats[0]   # (16, 16, 728)
    skip_8     = feats[1]   # (8,  8, 1024)
    bottleneck = feats[2]   # (4,  4, 2048)

    # Decoder: 4→8→16→32→64→128
    # --- 4x4 → 8x8 with skip_8 ---
    x = layers.Conv2DTranspose(512, (3,3), strides=2, padding="same", use_bias=False)(bottleneck)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Concatenate()([x, skip_8])
    x = layers.Conv2D(512, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)

    # --- 8x8 → 16x16 with skip_16 ---
    x = layers.Conv2DTranspose(256, (3,3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Concatenate()([x, skip_16])
    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)

    # --- 16x16 → 32x32 (no skip) ---
    x = layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)

    # --- 32x32 → 64x64 ---
    x = layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)

    # --- 64x64 → 128x128 → output ---
    x = layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(num_classes, (1,1), activation="softmax", name="output_mask")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet_xception")


model = build_unet_xception()
model.summary()

summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))
with open(os.path.join(OUT, "step4_model_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print("  saved: step4_model_summary.txt")

fig, ax = plt.subplots(figsize=(10, 7))
ax.axis("off")
ax.text(0.02, 0.98, "\n".join(summary_lines[:65]), transform=ax.transAxes,
        fontsize=6, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
fig.suptitle("Step 4 — U-Net Xception Architecture", fontsize=11)
fig.tight_layout()
savefig("step4_model_architecture.png", fig)


# ── Step 5: Train ─────────────────────────────────────────────────────────────
print("\n[Step 5] Training")

# MeanIoU in Keras 3 needs class indices, not softmax probabilities — wrap with argmax
class MeanIoUSparse(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", MeanIoUSparse(num_classes=NUM_CLASSES, name="mean_iou")],
)

checkpoint_path = os.path.join(OUT, "step5_best_model.keras")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_mean_iou", mode="max",
        save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_iou", mode="max", patience=8,
        restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=25,
    callbacks=callbacks,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ep = range(1, len(history.history["loss"]) + 1)

axes[0].plot(ep, history.history["loss"],         label="Train")
axes[0].plot(ep, history.history["val_loss"],     label="Val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(ep, history.history["accuracy"],     label="Train")
axes[1].plot(ep, history.history["val_accuracy"], label="Val")
axes[1].set_title("Pixel Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

axes[2].plot(ep, history.history["mean_iou"],     label="Train")
axes[2].plot(ep, history.history["val_mean_iou"], label="Val")
axes[2].set_title("Mean IoU"); axes[2].set_xlabel("Epoch"); axes[2].legend()

fig.suptitle("Step 5 — Training History", fontsize=11)
fig.tight_layout()
savefig("step5_training_history.png", fig)


# ── Step 6: Predict & Visualize ───────────────────────────────────────────────
print("\n[Step 6] Generating predictions on test data")

best_model = tf.keras.models.load_model(
    checkpoint_path, custom_objects={"MeanIoUSparse": MeanIoUSparse}
)
results    = best_model.evaluate(test_ds, verbose=0)
print(f"  Test loss:      {results[0]:.4f}")
print(f"  Test accuracy:  {results[1]:.4f}")
print(f"  Test mean IoU:  {results[2]:.4f}")

# Prediction grid — 4 samples
test_samples = list(test_ds.unbatch().take(4))
t_imgs  = np.stack([x[0].numpy() for x in test_samples])
t_masks = np.stack([x[1].numpy() for x in test_samples])
p_masks = np.argmax(best_model.predict(t_imgs, verbose=0), axis=-1)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    axes[0][i].imshow(t_imgs[i]);              axes[0][i].axis("off")
    axes[1][i].imshow(mask_to_rgb(t_masks[i])); axes[1][i].axis("off")
    axes[2][i].imshow(mask_to_rgb(p_masks[i])); axes[2][i].axis("off")

for row, label in enumerate(["Input Image", "Ground Truth", "Prediction"]):
    axes[row][0].set_ylabel(label, fontsize=10, rotation=0, labelpad=90, va="center")

handles = [plt.Rectangle((0,0),1,1, color=c/255, label=n)
           for c, n in zip(PALETTE, CLASS_NAMES)]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"Step 6 — Test Set Predictions  (Mean IoU = {results[2]:.3f})", fontsize=12)
fig.tight_layout()
savefig("step6_predictions.png", fig)

# Per-class IoU from confusion matrix
iou_metric = tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)
for img_b, mask_b in test_ds:
    pred_b = np.argmax(best_model.predict(img_b, verbose=0), axis=-1)
    iou_metric.update_state(mask_b, pred_b)

cm = iou_metric.get_weights()[0]
per_class_iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-7)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_NAMES, per_class_iou * 100, color=[c / 255 for c in PALETTE])
ax.axhline(results[2] * 100, color="black", linestyle="--", linewidth=1.2,
           label=f"Mean IoU = {results[2]*100:.1f}%")
ax.set_ylim(0, 105)
ax.set_ylabel("IoU (%)")
ax.set_title("Step 6 — Per-Class IoU Score")
ax.legend()
for bar, iou in zip(bars, per_class_iou):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{iou*100:.1f}%", ha="center", fontsize=10)
fig.tight_layout()
savefig("step6_iou_per_class.png", fig)

print(f"\nDone. Mean IoU: {results[2]*100:.1f}%")
print("All outputs saved to:", OUT)

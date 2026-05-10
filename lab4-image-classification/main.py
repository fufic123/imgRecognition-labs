import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report

OUT = os.path.join(os.path.dirname(__file__), "output")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def savefig(name, fig):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


# ── Step 1: Load & explore CIFAR-10 ──────────────────────────────────────────
print("[Step 1] Loading CIFAR-10 dataset")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
print(f"  train: {x_train.shape}  test: {x_test.shape}")

# Sample grid — 5×10, one row per class
fig, axes = plt.subplots(10, 5, figsize=(10, 20))
for cls in range(10):
    idx = np.where(y_train == cls)[0][:5]
    for col, i in enumerate(idx):
        axes[cls][col].imshow(x_train[i])
        axes[cls][col].axis("off")
        if col == 0:
            axes[cls][col].set_ylabel(CLASS_NAMES[cls], fontsize=8, rotation=0,
                                       labelpad=40, va="center")
fig.suptitle("CIFAR-10 — Sample Images (5 per class)", fontsize=12)
fig.tight_layout()
savefig("step1_dataset_samples.png", fig)


# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
print("\n[Step 2] Preprocessing — normalize + augmentation")

# Normalization: scale [0, 255] → [0.0, 1.0]
x_train_n = x_train.astype("float32") / 255.0
x_test_n  = x_test.astype("float32")  / 255.0

# Show normalization effect
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
for i in range(5):
    axes[0][i].imshow(x_train[i])
    axes[0][i].set_title(f"Raw  [{x_train[i].min()}–{x_train[i].max()}]", fontsize=8)
    axes[0][i].axis("off")
    axes[1][i].imshow(x_train_n[i])
    axes[1][i].set_title(f"Norm [{x_train_n[i].min():.2f}–{x_train_n[i].max():.2f}]", fontsize=8)
    axes[1][i].axis("off")
axes[0][0].set_ylabel("Before", fontsize=9)
axes[1][0].set_ylabel("After", fontsize=9)
fig.suptitle("Step 2 — Normalization", fontsize=11)
fig.tight_layout()
savefig("step2_normalization.png", fig)

# Data augmentation pipeline
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

# Show augmentation — same image 8 times
sample_img = tf.expand_dims(x_train_n[0], 0)
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
axes[0][0].imshow(x_train_n[0]); axes[0][0].set_title("Original"); axes[0][0].axis("off")
for idx, ax in enumerate(list(axes.flat)[1:]):
    aug = augment(sample_img, training=True)[0].numpy()
    ax.imshow(np.clip(aug, 0, 1))
    ax.set_title(f"Aug {idx+1}", fontsize=8)
    ax.axis("off")
fig.suptitle("Step 2 — Augmentation (rotation, flip, zoom, translate)", fontsize=11)
fig.tight_layout()
savefig("step2_augmentation_samples.png", fig)


# ── Step 3: Build CNN model ───────────────────────────────────────────────────
print("\n[Step 3] Building CNN model")

model = models.Sequential([
    # Augmentation (applied only during training)
    augment,

    # Block 1 — 'same' padding keeps spatial size
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2 — 'valid' padding reduces spatial size
    layers.Conv2D(64, (3, 3), padding="valid", activation="relu"),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="cifar10_cnn")

model.summary()

# Save architecture as text
summary_path = os.path.join(OUT, "step3_model_summary.txt")
with open(summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
print(f"  saved: step3_model_summary.txt")

# Architecture diagram as figure
fig, ax = plt.subplots(figsize=(9, 7))
ax.axis("off")
summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))
ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
        fontsize=6.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
fig.suptitle("CNN Model Architecture", fontsize=11)
fig.tight_layout()
savefig("step3_model_architecture.png", fig)


# ── Step 4: Train ─────────────────────────────────────────────────────────────
print("\n[Step 4] Training")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train_n, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, len(history.history["loss"]) + 1)

axes[0].plot(epochs_range, history.history["loss"],     label="Train loss")
axes[0].plot(epochs_range, history.history["val_loss"], label="Val loss")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(epochs_range, history.history["accuracy"],     label="Train acc")
axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val acc")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

fig.suptitle("Step 4 — Training History", fontsize=11)
fig.tight_layout()
savefig("step4_training_history.png", fig)


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
print("\n[Step 5] Evaluation")

test_loss, test_acc = model.evaluate(x_test_n, y_test, verbose=0)
print(f"  Test accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(x_test_n, verbose=0), axis=1)

# F1 scores
f1_macro    = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
f1_per_class = f1_score(y_test, y_pred, average=None)
print(f"  F1 macro:    {f1_macro:.4f}")
print(f"  F1 weighted: {f1_weighted:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Save classification report as text
report_path = os.path.join(OUT, "step5_classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Test accuracy : {test_acc:.4f}\n")
    f.write(f"F1 macro      : {f1_macro:.4f}\n")
    f.write(f"F1 weighted   : {f1_weighted:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print("  saved: step5_classification_report.txt")

# First 9 test images with true and predicted labels
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i])
    true_label = CLASS_NAMES[y_test[i]]
    pred_label = CLASS_NAMES[y_pred[i]]
    correct = y_test[i] == y_pred[i]
    color = "green" if correct else "red"
    ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                 color=color, fontsize=9)
    ax.axis("off")
fig.suptitle(f"Step 5 — First 9 Test Predictions  (green=correct, red=wrong)\n"
             f"Accuracy: {test_acc*100:.1f}%   F1 macro: {f1_macro*100:.1f}%", fontsize=11)
fig.tight_layout()
savefig("step5_test_predictions.png", fig)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title(f"Step 5 — Confusion Matrix  (acc={test_acc*100:.1f}%  F1={f1_macro*100:.1f}%)", fontsize=11)
fig.tight_layout()
savefig("step5_confusion_matrix.png", fig)

# Per-class F1 bar chart
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(CLASS_NAMES, f1_per_class * 100,
              color=["steelblue" if f >= 0.7 else "tomato" for f in f1_per_class])
ax.axhline(f1_macro * 100, color="black", linestyle="--", linewidth=1.2,
           label=f"Macro F1 = {f1_macro*100:.1f}%")
ax.set_ylim(0, 105)
ax.set_ylabel("F1 Score (%)")
ax.set_title("Step 5 — Per-Class F1 Score")
ax.legend()
for bar, f in zip(bars, f1_per_class):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{f*100:.1f}%", ha="center", fontsize=8)
fig.tight_layout()
savefig("step5_per_class_f1.png", fig)

print(f"\nDone. Accuracy: {test_acc*100:.1f}%  |  F1 macro: {f1_macro*100:.1f}%")
print("All outputs saved to:", OUT)

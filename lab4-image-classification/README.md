# Lab 4 — Image Classification and Object Detection

## Description

This lab introduces the principles of image classification using deep learning methods. Image classification is a widely applied task in both science and industry, enabling computers to automatically recognize and classify objects in images based on their content. Using TensorFlow, students learn to build and train neural networks for image classification tasks, covering data preparation, model architecture creation, training, and evaluation.

## Objective

To develop practical skills in building and training convolutional neural networks (CNNs) for image classification using the CIFAR-10 dataset.

---

## Step 1 — Dataset: CIFAR-10

**Goal:** Load and explore the CIFAR-10 dataset to understand its structure before building a model.

CIFAR-10 contains 60,000 color images (32×32 px) across 10 classes:

| Split | Images |
|---|---|
| Train | 50,000 |
| Test | 10,000 |
| Classes | 10 (1,000 test images each) |

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

Since each class has exactly 1,000 test samples, the dataset is perfectly balanced — making both accuracy and F1 score equally valid metrics here. F1 is still preferred because it separately captures precision and recall per class.

![Dataset samples](output/step1_dataset_samples.png)

---

## Step 2 — Data Preprocessing

**Goal:** Prepare images for training by normalizing pixel values and creating an augmentation pipeline.

### Normalization

```python
x_train_n = x_train.astype("float32") / 255.0
x_test_n  = x_test.astype("float32")  / 255.0
```

Scales pixel values from [0, 255] to [0.0, 1.0]. This keeps weight gradients in a stable range during backpropagation and speeds up convergence.

![Normalization](output/step2_normalization.png)

### Augmentation

```python
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])
```

Applied **inside the model** during training only — each epoch the same image is seen in a slightly different form, reducing overfitting without increasing dataset size on disk.

| Transform | Range | Purpose |
|---|---|---|
| RandomFlip | horizontal | Mirror symmetry |
| RandomRotation | ±15% of 360° | Orientation variance |
| RandomZoom | ±15% | Scale variance |
| RandomTranslation | ±10% | Position variance |

![Augmentation samples](output/step2_augmentation_samples.png)

---

## Step 3 — CNN Architecture

**Goal:** Build a convolutional neural network with dropout regularization, different padding strategies, and multiple dense layers.

```
Input (32×32×3)
│
├─ Augmentation layer (training only)
│
├─ Block 1: Conv2D(32, 3×3, padding=same) → Conv2D(32, 3×3, same) → MaxPool → Dropout(0.25)
├─ Block 2: Conv2D(64, 3×3, padding=valid) → Conv2D(64, 3×3, same) → MaxPool → Dropout(0.25)
├─ Block 3: Conv2D(128, 3×3, padding=same) → MaxPool → Dropout(0.4)
│
├─ Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Dense(10, softmax)
```

**Padding:**
- `same` — output has the same spatial size as input; preserves edge features
- `valid` — no padding, output shrinks; reduces spatial dimensions more aggressively

**Dropout rates:** 0.25 after conv blocks (light), 0.5 after the large dense layer (aggressive) — higher dropout near the classifier head is a common pattern to prevent co-adaptation of neurons.

**Total parameters: 468,778 (~1.8 MB)**

![Model architecture](output/step3_model_architecture.png)

---

## Step 4 — Training

**Goal:** Train the model using the Adam optimizer and sparse categorical crossentropy loss for 20 epochs.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(x_train_n, y_train, epochs=20, batch_size=64, validation_split=0.1)
```

**Loss function:** `sparse_categorical_crossentropy` — used when labels are integers (not one-hot encoded). Equivalent to categorical crossentropy but avoids the overhead of converting labels.

**Optimizer:** Adam — adaptive learning rate per parameter, combining momentum and RMSProp. Converges faster than plain SGD on most vision tasks.

![Training history](output/step4_training_history.png)

---

## Step 5 — Evaluation

**Goal:** Evaluate model performance using accuracy and F1 score, display the first 9 test predictions, and analyze per-class results.

### Why F1 over accuracy?

Accuracy shows the overall fraction correct but hides imbalance between precision and recall. F1 is the harmonic mean of precision and recall — a class with high precision but poor recall (model is overly conservative) will show a low F1 even if accuracy looks acceptable.

```python
f1_macro    = f1_score(y_test, y_pred, average="macro")    # unweighted mean across classes
f1_weighted = f1_score(y_test, y_pred, average="weighted")  # weighted by class support
```

### Results

| Metric | Score |
|---|---|
| Test accuracy | 60.5% |
| F1 macro | 58.8% |
| F1 weighted | 58.8% |

The gap between accuracy and F1 macro reveals that some classes (e.g. `cat`, `bird`) pull down the average — they have reasonable precision but low recall, meaning the model misses many true instances of those classes.

### Per-class breakdown

| Class | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 0.71 | 0.67 | 0.69 |
| automobile | 0.66 | 0.90 | 0.76 |
| bird | 0.74 | 0.28 | 0.41 |
| cat | 0.51 | 0.30 | 0.38 |
| deer | 0.70 | 0.35 | 0.47 |
| dog | 0.67 | 0.47 | 0.55 |
| frog | 0.46 | 0.78 | 0.58 |
| horse | 0.63 | 0.74 | 0.68 |
| ship | 0.77 | 0.74 | 0.76 |
| truck | 0.48 | 0.82 | 0.61 |

`automobile` and `ship` perform best (F1 ≥ 0.76). `cat` and `bird` are the hardest — visually similar to `dog` and `deer` respectively, which causes frequent confusions visible in the confusion matrix.

### First 9 test predictions

![Test predictions](output/step5_test_predictions.png)

### Per-class F1

![Per-class F1](output/step5_per_class_f1.png)

### Confusion Matrix

Rows = true class, columns = predicted class. Strong diagonal = correct predictions. Off-diagonal entries reveal systematic confusions (e.g. `cat` → `dog`, `deer` → `horse`).

![Confusion matrix](output/step5_confusion_matrix.png)

---

## Output Files

| File | Description |
|---|---|
| `step1_dataset_samples.png` | 5 samples per class grid |
| `step2_normalization.png` | Raw vs normalized pixel values |
| `step2_augmentation_samples.png` | Augmentation variations of one image |
| `step3_model_architecture.png` | Model layer summary |
| `step3_model_summary.txt` | Full model summary text |
| `step4_training_history.png` | Loss and accuracy curves over 20 epochs |
| `step5_test_predictions.png` | First 9 test images with true/predicted labels |
| `step5_per_class_f1.png` | Per-class F1 score bar chart |
| `step5_confusion_matrix.png` | 10×10 confusion matrix |
| `step5_classification_report.txt` | Full precision/recall/F1 report |

## Summary

| Step | Key operation | Tool |
|---|---|---|
| Data loading | CIFAR-10 (50k train / 10k test) | `tf.keras.datasets.cifar10` |
| Normalization | Scale [0,255] → [0,1] | NumPy |
| Augmentation | Flip, rotation, zoom, translate | `tf.keras.layers.Random*` |
| Model | 3 conv blocks + 2 dense layers | `tf.keras.Sequential` |
| Training | Adam + sparse_categorical_crossentropy | `model.fit()` |
| Evaluation | Accuracy + F1 macro/weighted + confusion matrix | `sklearn.metrics` |

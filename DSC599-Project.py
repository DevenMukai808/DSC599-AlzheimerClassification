import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Rescaling
)
from tensorflow.keras.callbacks import EarlyStopping


# Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("ninadaithal/imagesoasis")
print("Downloaded dataset to:", dataset_path)

expected_classes = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia",
]

def find_image_directory(root_path):
    """Find a directory containing the expected class folders."""
    for current_root, directories, _ in os.walk(root_path):
        normalized = {name.strip().lower() for name in directories}
        if all(name.lower() in normalized for name in expected_classes):
            return current_root

    
    for current_root, directories, _ in os.walk(root_path):
        normalized = {name.strip().lower() for name in directories}
        if (
            any("mild" in name and "very" not in name for name in normalized)
            and any("moderate" in name for name in normalized)
            and any("non" in name and "dement" in name for name in normalized)
            and any("very" in name and "mild" in name for name in normalized)
        ):
            return current_root

    raise FileNotFoundError(
        "Could not find the four class folders. Inspect dataset_path and update "
        "expected_classes to match the downloaded folder names."
    )

data_dir = find_image_directory(dataset_path)
print("Image folder:", data_dir)
print("Subfolders:", sorted(os.listdir(data_dir)))

folder_names = set(os.listdir(data_dir))
class_lookup = {name.strip().lower(): name for name in folder_names}

class_names = []
for wanted in expected_classes:
    actual = class_lookup.get(wanted.lower())
    if actual is None:
        # Handle common capitalization variations.
        matches = [name for name in folder_names if name.strip().lower() == wanted.lower()]
        if not matches:
            raise FileNotFoundError(
                f"Expected class folder '{wanted}' was not found. "
                f"Found folders: {sorted(folder_names)}"
            )
        actual = matches[0]
    class_names.append(actual)

classes = class_names
n_classes = len(classes)


# Load images, create training, validation, and test sets
seed = 42
image_size = (224, 224)
batch_size = 32

# Reserve 30% of images for validation + testing, then split that holdout
# approximately in half. This creates roughly 70% train / 15% validation /
# 15% test. Note: this is an image-level split, not a patient-level split.
training_dataset, holdout_dataset = image_dataset_from_directory(
    data_dir,
    class_names=classes,
    image_size=image_size,
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    validation_split=0.30,
    subset="both",
    label_mode="categorical",
)

holdout_batches = tf.data.experimental.cardinality(holdout_dataset).numpy()
if holdout_batches < 2:
    raise ValueError("Not enough holdout batches to create validation and test sets.")

split_batch = holdout_batches // 2
validation_dataset = holdout_dataset.take(split_batch)
testing_dataset = holdout_dataset.skip(split_batch)

AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
testing_dataset = testing_dataset.prefetch(buffer_size=AUTOTUNE)

print("Class order:", classes)
print("Number of classes:", n_classes)
print("Training batches:", tf.data.experimental.cardinality(training_dataset).numpy())
print("Validation batches:", tf.data.experimental.cardinality(validation_dataset).numpy())
print("Testing batches:", tf.data.experimental.cardinality(testing_dataset).numpy())


# Plot class distribution
def count_classes(dataset):
    class_counts = np.zeros(len(classes), dtype=int)
    for _, labels in dataset:
        class_indices = np.argmax(labels.numpy(), axis=1)
        for idx in class_indices:
            class_counts[idx] += 1
    return class_counts

train_counts = count_classes(training_dataset)
val_counts = count_classes(validation_dataset)
test_counts = count_classes(testing_dataset)

bar_width = 0.25
positions = np.arange(len(classes))

plt.figure(figsize=(12, 6))
plt.bar(positions - bar_width, train_counts, width=bar_width, label="Train")
plt.bar(positions, val_counts, width=bar_width, label="Validation")
plt.bar(positions + bar_width, test_counts, width=bar_width, label="Test")
plt.xlabel("Dementia category")
plt.ylabel("Number of images")
plt.title("Class Distribution Across Data Splits")
plt.xticks(positions, classes, rotation=20, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Visualize sample MRI images
images, labels = next(iter(training_dataset))
num_images = min(15, len(images))

plt.figure(figsize=(12, 9))
for i in range(num_images):
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    label_idx = np.argmax(labels[i].numpy())
    plt.title(classes[label_idx], fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()


# Create CNN model
model = Sequential([
    Rescaling(1.0 / 255, input_shape=(224, 224, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax"),
])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

model.summary()


# Compile and train model
# Limit, takes too much time
epochs = 25

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Uncomment later, takes too long
"""
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stop],
)
"""

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=5,
    steps_per_epoch=20,
    validation_steps=3,
    callbacks=[early_stop]
)


# Plot training results
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Evaluate on test set
test_loss, test_accuracy = model.evaluate(testing_dataset)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")



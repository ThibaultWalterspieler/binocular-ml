import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

training_dir = "data/PeopleWithGlassesDataset/train"

with_glasses_length = len(os.listdir(training_dir + "/with_glasses"))
without_glasses_length = len(os.listdir(training_dir + "/without_glasses"))

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1, 0.2),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    ]
)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=20,
    label_mode="binary",
).map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=20,
    label_mode="binary",
).map(
    lambda x, y: (data_augmentation(x, training=False), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


model = keras.Sequential(
    [
        layers.Input(shape=(128, 128, 3), name="input"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
        layers.MaxPooling2D(2, 2, name="pool1"),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv2"),
        layers.MaxPooling2D(2, 2, name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", "precision", "recall", "AUC"],
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

without_glasses_weight = (
    (1 / with_glasses_length) * (with_glasses_length + without_glasses_length) / 2.0
)
with_glasses_weight = (
    (1 / without_glasses_length) * (with_glasses_length + without_glasses_length) / 2.0
)
class_weight = {0: without_glasses_weight, 1: with_glasses_weight}

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    class_weight=class_weight,
    callbacks=[early_stopping],
    verbose=1,
)

model.save("binocular_model.keras")

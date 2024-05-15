import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

TRAINING_DIR = "data/PeopleWithGlassesDataset/train"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 20
VALIDATION_SPLIT = 0.2
SEED = 123


def calculate_class_weights():
    with_glasses_length = len(os.listdir(TRAINING_DIR + "/with_glasses"))
    without_glasses_length = len(os.listdir(TRAINING_DIR + "/without_glasses"))
    total_length = with_glasses_length + without_glasses_length
    without_glasses_weight = (1 / without_glasses_length) * (total_length / 2.0)
    with_glasses_weight = (1 / with_glasses_length) * (total_length / 2.0)
    return {0: without_glasses_weight, 1: with_glasses_weight}


def create_data_augmentation():
    return keras.Sequential(
        [
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1, 0.2),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        ]
    )


def load_and_preprocess_dataset(data_augmentation):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAINING_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    ).map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAINING_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    ).map(
        lambda x, y: (data_augmentation(x, training=False), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_dataset = (
        train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    validation_dataset = validation_dataset.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset


def create_model():
    return keras.Sequential(
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


def compile_model(model):
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "precision", "recall", "AUC"],
    )


def create_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]


def train_model(model, train_dataset, validation_dataset, class_weight, callbacks):
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=validation_dataset,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model(model, model_path):
    model.save(model_path)


def main():
    class_weight = calculate_class_weights()
    data_augmentation = create_data_augmentation()
    train_dataset, validation_dataset = load_and_preprocess_dataset(data_augmentation)
    model = create_model()
    compile_model(model)
    callbacks = create_callbacks()
    history = train_model(
        model, train_dataset, validation_dataset, class_weight, callbacks
    )
    save_model(model, "binocular_model.keras")


if __name__ == "__main__":
    main()

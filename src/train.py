import tensorflow as tf

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
IMG_SIZE = 180


def create_model():
    resize_and_rescale = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
            tf.keras.layers.Rescaling(1.0 / 255),
        ]
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ]
    )

    model = tf.keras.Sequential(
        [
            resize_and_rescale,
            data_augmentation,
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory="data/PeopleWithGlassesDataset/train",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory="data/PeopleWithGlassesDataset/train",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def compile_and_train(model, train_ds, val_ds):
    model.compile(
        optimizer="adam",
        loss=tf.losses.BinaryCrossentropy(),
        metrics=["accuracy", "precision", "recall", "AUC"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[
            early_stopping,
        ],
    )
    return history


def main():
    model = create_model()
    train_ds, val_ds = load_data()
    history = compile_and_train(model, train_ds, val_ds)
    model.save("models/next.keras")


if __name__ == "__main__":
    main()

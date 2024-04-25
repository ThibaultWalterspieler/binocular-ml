import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

TRAINING_DIR = "data/PeopleWithGlassesDataset/train"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 20
AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(training_dir, image_size, batch_size):
    data_augmentation = keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1, 0.2),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    ])

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    ).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    ).map(lambda x, y: (data_augmentation(x, training=False), y), num_parallel_calls=AUTOTUNE)

    return train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE), \
        validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input'),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        layers.MaxPooling2D(2, 2, name='pool1'),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D(2, 2, name='pool2'),
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'AUC'])
    return model


def compute_class_weights(training_dir):
    with_glasses_length = len(os.listdir(
        os.path.join(training_dir, 'with_glasses')))
    without_glasses_length = len(os.listdir(
        os.path.join(training_dir, 'without_glasses')))
    total = with_glasses_length + without_glasses_length
    return {
        0: (1 / with_glasses_length) * (total) / 2.0,
        1: (1 / without_glasses_length) * (total) / 2.0
    }


def train_model(model, train_dataset, validation_dataset, class_weights, epochs=100):
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    return history


def main():
    train_dataset, validation_dataset = load_datasets(
        TRAINING_DIR, IMAGE_SIZE, BATCH_SIZE)
    model = create_model((128, 128, 3))
    class_weights = compute_class_weights(TRAINING_DIR)
    history = train_model(model, train_dataset,
                          validation_dataset, class_weights)
    model.save('models/binocular_model.keras')
    model.save('models/binocular_model.h5')


if __name__ == "__main__":
    main()

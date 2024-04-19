import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'datasets/PeopleWithGlassesDataset'
with_glasses_length = len(os.listdir(
    "datasets/PeopleWithGlassesDataset/with_glasses"))
without_glasses_length = len(os.listdir(
    "datasets/PeopleWithGlassesDataset/without_glasses"))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='binary',
    subset='validation'
)

model = tf.keras.models.Sequential([
    Input(shape=(64, 64, 3)),  # Explicit input layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


without_glasses_weight = (1 / with_glasses_length) * \
    (without_glasses_length + with_glasses_length) / 2.0
with_glasses_weight = (1 / without_glasses_length) * \
    (without_glasses_length + with_glasses_length) / 2.0

class_weight = {0: without_glasses_weight, 1: with_glasses_weight}

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples,
    class_weight=class_weight,
    callbacks=[early_stopping]
)

model.save('binocular.keras')
print("Model saved successfully!")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

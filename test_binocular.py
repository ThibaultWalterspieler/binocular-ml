import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

test_dir = 'test/PeopleWithGlassesDataset'

model = keras.models.load_model('binocular_model.keras')

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=20,
    shuffle=False,
    label_mode='binary'
)

for images, labels in test_dataset:
    predictions = model.predict(images)
    for i in range(len(predictions)):
        print(f'Prediction: {predictions[i][0]:.4f}, Actual: {labels[i]}')
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(f'Prediction: {predictions[i][0]:.4f}, Actual: {labels[i]}')
        plt.show()

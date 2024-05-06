import os

import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow import keras

test_dataset_path = os.path.join("data", "PeopleWithGlassesDataset", "test")
model_path = os.path.join("models", "next.keras")

MODEL = keras.models.load_model(model_path)
detector = MTCNN()

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dataset_path,
    image_size=(128, 128),
    batch_size=20,
    shuffle=False,
    label_mode="binary",
)

for images, labels in test_dataset:
    images_np = images.numpy()

    for i, image in enumerate(images_np):
        result = detector.detect_faces(image)
        if result:
            face = result[0]["box"]
            face_x, face_y, face_width, face_height = face
            face_x, face_y = abs(face_x), abs(face_y)
            face_frame = image[
                face_y : face_y + face_height, face_x : face_x + face_width
            ]
            face_frame = tf.image.resize(face_frame, [128, 128])
            face_frame = tf.expand_dims(face_frame / 255.0, axis=0)

            prediction = MODEL.predict(face_frame)
            print(prediction)
            not_wearing_glasses = prediction > 0.5
            text = "Without glasses" if not_wearing_glasses else "With glasses"
            text = f"{text} ({prediction[0][0]:.2f})"

            plt.imshow(face_frame[0])
            plt.title(text)
            plt.show()
        else:
            print("No face detected in image.")

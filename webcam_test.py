import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

model = tf.keras.models.load_model('binocular.keras')

detector = MTCNN()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    faces = detector.detect_faces(frame)
    for face in faces:
        face_x, face_y, face_width, face_height = face['box']
        face_x, face_y = abs(face_x), abs(face_y)
        face_frame = frame[face_y:face_y+face_height, face_x:face_x+face_width]

        face_frame = cv2.resize(face_frame, (64, 64))
        face_frame = face_frame.astype('float32')
        face_frame /= 255.0
        face_frame = np.expand_dims(face_frame, axis=0)

        prediction = model.predict(face_frame)
        print(prediction)
        predicted_class = int(prediction > 0.5)
        text = "No glasses" if predicted_class == 1 else "Wearing glasses"
        text = f"{text} ({prediction[0][0]:.2f})"
        color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)

        cv2.putText(frame, text, (face_x, face_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (face_x, face_y), (face_x +
                      face_width, face_y + face_height), color, 2)

    cv2.rectangle(frame, (0, frame.shape[0] - 35),
                  (frame.shape[1], frame.shape[0]), (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, "Press 'q' to quit",
                (10,
                 frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Binocle detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

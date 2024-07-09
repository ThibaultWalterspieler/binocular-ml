from tensorflow import keras
from tensorflowjs import converters

model_path = "models/basic_training_with_130k_pic_20240612-182711.keras"
model = keras.models.load_model(model_path)
model.summary()
converters.save_keras_model(model, "tfjs_model")

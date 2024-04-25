import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model = load_model('binocular_model.h5')
model.summary()
tfjs.converters.save_keras_model(model, "test")

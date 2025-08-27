import numpy as np
'''import tensorflow as tf
load_model = tf.keras.models.load_model'''
from keras.models import load_model   # ✅ cleaner import


def load_sign_model(model_path="model/sign_model.h5", labels_path="model/labels.npy"):
    model = load_model(model_path)
    labels = np.load(labels_path, allow_pickle=True)
    return model, labels

def predict_sign(model, labels, img):
    prediction = model.predict(img)[0]
    label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return label, confidence

"""Model loading and preprocessing for the brain tumour classifier.

Kept separate from the Streamlit UI in app.py so it can be tested on its own.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "best_cnnmodel_1.h5"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def load_model(path=MODEL_PATH):
    """Load the trained Keras CNN."""
    import tensorflow as tf  # imported here so preprocessing works without TensorFlow

    return tf.keras.models.load_model(path)


def preprocess(image):
    """Convert a PIL image into the array the model expects.

    The model was trained on 150x150 images read with OpenCV (BGR channel
    order, pixel values 0-255), so the image is resized, converted to RGB,
    then flipped to BGR. Returns a float32 array of shape (150, 150, 3).
    """
    image = ImageOps.fit(image.convert("RGB"), IMG_SIZE, Image.LANCZOS)
    return np.asarray(image, dtype="float32")[..., ::-1]


def predict(image, model):
    """Return one probability per class in CLASS_NAMES for a PIL image."""
    batch = preprocess(image)[np.newaxis, ...]
    return model.predict(batch, verbose=0)[0]

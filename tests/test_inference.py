import numpy as np
import pytest
from PIL import Image

import inference


def test_preprocess_shape_and_dtype():
    image = Image.new("RGB", (300, 200), (10, 20, 30))
    array = inference.preprocess(image)
    assert array.shape == (150, 150, 3)
    assert array.dtype == np.float32


def test_preprocess_uses_bgr_order_and_0_255_range():
    red = Image.new("RGB", (150, 150), (255, 0, 0))
    array = inference.preprocess(red)
    # Training images were read with OpenCV, so red must end up in the last channel.
    assert np.allclose(array[..., 0], 0)
    assert np.allclose(array[..., 2], 255)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_preprocess_accepts_other_image_modes(mode):
    image = Image.new(mode, (64, 64))
    assert inference.preprocess(image).shape == (150, 150, 3)


def test_model_outputs_one_probability_per_class():
    pytest.importorskip("tensorflow")
    if not inference.MODEL_PATH.exists():
        pytest.skip("model file not present")
    model = inference.load_model()
    probabilities = inference.predict(Image.new("RGB", (150, 150), (128, 128, 128)), model)
    assert probabilities.shape == (len(inference.CLASS_NAMES),)
    assert np.all(probabilities >= 0)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-3)

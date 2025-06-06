import os

import numpy as np
import pytest
from PIL import Image

from model import ImagePreprocessor, OnnxModel

# Constants for testing
TEST_IMAGES_DIR = "imgs"  # Relative to workspace root
TENCH_IMAGE = "n01440764_tench.jpeg"  # Label 0
MUD_TURTLE_IMAGE = "n01667114_mud_turtle.JPEG"  # Label 35
MODEL_PATH = "models/model.onnx"  # Relative to workspace root


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    return OnnxModel(MODEL_PATH)


@pytest.fixture
def test_images():
    images = {}
    for img_name in [TENCH_IMAGE, MUD_TURTLE_IMAGE]:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Test image not found at {img_path}")
            pytest.skip(f"Test image not found at {img_path}")
        images[img_name] = Image.open(img_path)
    return images


def test_preprocessor_initialization():
    preprocessor = ImagePreprocessor()
    assert preprocessor.size == (224, 224)
    assert preprocessor.transform is not None


def test_preprocessor_invalid_input():
    preprocessor = ImagePreprocessor()
    with pytest.raises(TypeError):
        preprocessor.preprocess("not an image")


def test_preprocessor_output_shape(preprocessor, test_images):
    for img in test_images.values():
        output = preprocessor.preprocess(img)
        assert isinstance(output, np.ndarray)
        assert output.shape == (1, 3, 224, 224)  # (batch_size, channels, height, width)
        assert output.dtype == np.float32


def test_model_initialization():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}")
    model = OnnxModel(MODEL_PATH)
    assert model.session is not None
    assert model.input_name is not None
    assert model.output_name is not None


def test_model_prediction_shape(model, preprocessor, test_images):
    for img in test_images.values():
        output = model.predict_with_preprocessing(img, preprocessor)
        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 2  # (batch_size, num_classes)
        assert output.shape[0] == 1  # batch size of 1


def test_model_predictions(model, preprocessor, test_images):
    # Test tench image (label 0)
    tench_output = model.predict_with_preprocessing(
        test_images[TENCH_IMAGE], preprocessor
    )
    tench_prediction = np.argmax(tench_output[0])
    assert tench_prediction == 0, f"Expected label 0 (tench), got {tench_prediction}"

    # Test mud turtle image (label 35)
    turtle_output = model.predict_with_preprocessing(
        test_images[MUD_TURTLE_IMAGE], preprocessor
    )
    turtle_prediction = np.argmax(turtle_output[0])
    assert turtle_prediction == 35, (
        f"Expected label 35 (mud turtle), got {turtle_prediction}"
    )


def test_full_pipeline(model, preprocessor, test_images):
    for img_name, img in test_images.items():
        # Test the full pipeline
        preprocessed = preprocessor.preprocess(img)
        raw_output = model.predict(preprocessed)
        pipeline_output = model.predict_with_preprocessing(img, preprocessor)

        # Verify that both methods give the same results
        np.testing.assert_array_almost_equal(raw_output, pipeline_output)

        # Apply softmax to get probabilities
        exp_output = np.exp(pipeline_output[0])
        probabilities = exp_output / np.sum(exp_output)

        # Verify the prediction is reasonable (softmax probabilities sum to ~1)
        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-5)
        assert np.all(probabilities >= 0)  # All probabilities should be non-negative
        assert np.all(probabilities <= 1)  # All probabilities should be <= 1


def test_preprocessor_grayscale_image(preprocessor):
    # Create a grayscale image (L mode)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L"
    )
    output = preprocessor.preprocess(img)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 3, 224, 224)  # Should be converted to RGB


def test_preprocessor_rgba_image(preprocessor):
    # Create an RGBA image
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8), mode="RGBA"
    )
    output = preprocessor.preprocess(img)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 3, 224, 224)  # Should be converted to RGB


def test_preprocessor_palette_image(preprocessor):
    # Create a palette image (P mode)
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="P"
    )
    output = preprocessor.preprocess(img)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 3, 224, 224)  # Should be converted to RGB


def test_preprocessor_invalid_image(preprocessor):
    # Pass a completely invalid input (not a PIL image)
    with pytest.raises(TypeError):
        preprocessor.preprocess(12345)
    with pytest.raises(TypeError):
        preprocessor.preprocess(None)
    with pytest.raises(TypeError):
        preprocessor.preprocess(np.zeros((224, 224, 3)))

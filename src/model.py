import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # RGB means
                    std=[0.229, 0.224, 0.225],  # RGB stds
                ),
            ]
        )

    def preprocess(self, img: Image.Image) -> np.ndarray:
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image")
        img_tensor = self.transform(img)
        return img_tensor.unsqueeze(0).numpy()


class OnnxModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: input_data})[0]

    def predict_with_preprocessing(
        self, img: Image.Image, preprocessor: ImagePreprocessor
    ) -> np.ndarray:
        preprocessed = preprocessor.preprocess(img)
        return self.predict(preprocessed)

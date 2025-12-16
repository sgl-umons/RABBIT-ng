from importlib.resources import files
import logging

import numpy as np
import onnxruntime
from pandas import DataFrame
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """
    Abstract base class for BIMBAS model predictors. Subclasses must implement
    the `_load_model` and `predict` methods.
    An example subclass is `ONNXPredictor`, which uses ONNX Runtime to load and
    run inference on an ONNX model.

    Attributes:
        model_path (str): Path to the model file.
        model: Loaded model object.

    Methods:
        _load_model(): Abstract method to load the model from the specified path
        predict(features: DataFrame) -> tuple[str, float]:
            Abstract method to predict contributor type and confidence score.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, features: DataFrame) -> tuple[str, float]:
        """
        Predict if a contributor is a bot or a human using the BIMBAS model.

        Parameters:
            features: A Dataframe  with the features of the contributor.

        Returns:
            contributor_type (str) - type of contributor determined based on
                probability ('Bot' or 'Human')
            confidence (float) - confidence score of the determined type
                (value between 0.0 and 1.0)
        """
        pass


class ONNXPredictor(Predictor):
    def __init__(self, model_path: str | None = None):
        self.input_name = None
        self.output_name = None
        super().__init__(
            model_path
            if model_path
            else str(files("rabbit").joinpath("resources", "models", "bimbas.onnx"))
        )

    def _load_model(self):
        try:
            self.model = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[1].name

            logger.debug(f"Model ONNX loaded successfully from {self.model_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}: {e}"
            ) from e

    def predict(self, features: DataFrame) -> tuple[str, float]:
        input_data = features.values.astype("float32")

        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")
        # Run inference
        outputs = self.model.run([self.output_name], {self.input_name: input_data})

        probability_value = np.float64(outputs[0][0][1])

        contributor_type = "Bot" if probability_value >= 0.5 else "Human"
        confidence = (abs(probability_value - 0.5) * 2).round(3)

        return contributor_type, confidence

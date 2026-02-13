"""
Machine learning model predictors for RABBIT bot detection.

This module provides an abstract interface for bot classification models and
an ONNX Runtime implementation that loads and runs the pre-trained BIMBAS model.
"""

from importlib.resources import files
import logging

import numpy as np
import onnxruntime
from pandas import DataFrame
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """
    Abstract base class for bot detection model predictors.

    This class defines the interface for loading and running machine learning
    models that classify GitHub contributors as bots or humans based on
    behavioral features.

    Subclasses must implement model loading and prediction logic specific to
    their framework (e.g., ONNX, scikit-learn, PyTorch).

    Args:
        model_path: Path to the model file. If None, subclasses should use
            a default model path.

    Attributes:
        model_path: Path to the loaded model file.
        model: The loaded model object (type depends on subclass).

    Example:
        >>> predictor = ONNXPredictor()
        >>> features = DataFrame([[300, 250, ...]], columns=["NA", "NT", ...])
        >>> user_type, confidence = predictor.predict(features)
        >>> print(f"{user_type}: {confidence}")
        Human: 0.872
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
    """
    ONNX Runtime implementation of the BIMBAS bot detection model.

    This predictor loads a pre-trained ONNX model and performs inference
    using ONNX Runtime with CPU execution.

    Args:
        model_path: Path to the ONNX model file. If None, uses the bundled
            BIMBAS model from package resources.

    Attributes:
        model_path: Path to the ONNX model file.
        model: ONNX Runtime InferenceSession object.

    Raises:
        RuntimeError: If the model file cannot be loaded or is invalid.

    Example:
        >>> predictor = ONNXPredictor()
        >>> features = DataFrame([[42, 15, ...]], columns=["NA", "NT", ...])
        >>> user_type, confidence = predictor.predict(features)
        >>> print(f"{user_type}: {confidence}")
        Bot: 0.923
    """

    def __init__(self, model_path: str | None = None):
        self._input_name = None
        self._output_name = None
        super().__init__(
            model_path
            if model_path
            else str(files("rabbit_ng").joinpath("resources", "models", "bimbas.onnx"))
        )

    def _load_model(self):
        try:
            self.model = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self.model.get_inputs()[0].name
            self._output_name = self.model.get_outputs()[1].name

            logger.debug(f"Model ONNX loaded successfully from {self.model_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}: {e}"
            ) from e

    def predict(self, features: DataFrame) -> tuple[str, float]:
        """
        Predict contributor type using the BIMBAS ONNX model.

        The model outputs a probability P(bot). The contributor is classified
        as "Bot" if P >= 0.5, otherwise "Human". Confidence is calculated as
        2 * |P - 0.5|, representing how far the prediction is from the
        decision boundary.

        Args:
            features: DataFrame with one row containing all 38 behavioral
                features. Must have columns matching FEATURE_NAMES.

        Returns:
            Tuple of (contributor_type, confidence):
                - contributor_type: "Bot" or "Human"
                - confidence: Score from 0.0 to 1.0 (higher = more certain)

        Raises:
            RuntimeError: If the model is not loaded or inference fails.

        Example:
            >>> predictor = ONNXPredictor()
            >>> features = DataFrame([[42, 15, ...]], columns=["NA", "NT", ...])
            >>> user_type, confidence = predictor.predict(features)
            >>> print(f"{user_type}: {confidence}")
            Bot: 0.923
        """
        input_data = features.values.astype("float32")

        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")
        # Run inference
        outputs = self.model.run([self._output_name], {self._input_name: input_data})

        probability_value = np.float64(outputs[0][0][1])

        contributor_type = "Bot" if probability_value >= 0.5 else "Human"
        confidence = (abs(probability_value - 0.5) * 2).round(3)

        return contributor_type, confidence

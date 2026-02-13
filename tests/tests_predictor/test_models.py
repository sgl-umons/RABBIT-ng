import pytest

from rabbit_ng.predictor.models import ONNXPredictor
import os
import pandas as pd


class TestPredictor:
    @pytest.fixture
    def bot_features(self):
        """
        Fixture to load sample features from a CSV file.
        """
        bot_features_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "bot_features.csv"
        )
        features_df = pd.read_csv(bot_features_path)

        features = [
            "NA",
            "NT",
            "NOR",
            "ORR",
            "DCA_mean",
            "DCA_median",
            "DCA_std",
            "DCA_gini",
            "NAR_mean",
            "NAR_median",
            "NAR_gini",
            "NAR_IQR",
            "NTR_mean",
            "NTR_median",
            "NTR_std",
            "NTR_gini",
            "NCAR_mean",
            "NCAR_std",
            "NCAR_IQR",
            "DCAR_mean",
            "DCAR_median",
            "DCAR_std",
            "DCAR_IQR",
            "DAAR_mean",
            "DAAR_median",
            "DAAR_std",
            "DAAR_gini",
            "DAAR_IQR",
            "DCAT_mean",
            "DCAT_median",
            "DCAT_std",
            "DCAT_gini",
            "DCAT_IQR",
            "NAT_mean",
            "NAT_median",
            "NAT_std",
            "NAT_gini",
            "NAT_IQR",
        ]
        return features_df[features]

    def test_load_model_onnx(self):
        """
        Test loading the ONNX model.
        """
        predictor = ONNXPredictor()
        assert predictor.model is not None
        assert predictor._input_name is not None
        assert predictor._output_name is not None

    def test_fail_load_model_onnx(self):
        """
        Test failure to load an invalid ONNX model.
        """
        invalid_model_path = "invalid_model.onnx"
        with pytest.raises(RuntimeError) as excinfo:
            ONNXPredictor(model_path=invalid_model_path)
        assert "Failed to load model" in str(excinfo.value)

    def test_predict_onnx(self, bot_features):
        """
        Test the predict method of the ONNXPredictor with sample features.
        """
        predictor = ONNXPredictor()
        prediction, confidence = predictor.predict(bot_features)
        assert prediction == "Bot"
        assert confidence > 0

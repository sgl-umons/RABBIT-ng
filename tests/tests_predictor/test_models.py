def test_load_model():
    """
    As scikit_learn models are pickled objects, this test ensures that the model can be loaded correctly.
    """
    from rabbit.predictor.models import __load_model

    model = __load_model()
    assert model is not None


def test_predict_contributor():
    """
    Test the predict_contributor function with a sample bot feature set.
    """
    from rabbit.predictor.models import predict_contributor
    import os
    import pandas as pd

    # Sample features for testing
    bot_features_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "bot_features.csv"
    )
    sample_bot_features = pd.read_csv(bot_features_path)

    prediction, confidence = predict_contributor(sample_bot_features)
    assert prediction == "Bot"
    assert confidence == 0.882

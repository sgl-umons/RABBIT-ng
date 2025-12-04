import warnings
from importlib.resources import files

import joblib


def predict_contributor(features, model_path=None):
    """
    Predict if a contributor is a bot or not with a given model (BIMBAS architecture).

    Parameters:
        features: A DataFrame with the features of the contributor.
        model_path: The model to use to predict the type of contributor.

    Returns:
        contributor_type (str) - type of contributor determined based on
            probability ('Bot' or 'Human')
        confidence (float) - confidence score of the determined type
            (value between 0.0 and 1.0)
    """

    model = __load_model(model_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        probability_value = model.predict_proba(features)[0][1]

    contributor_type = "Bot" if probability_value >= 0.5 else "Human"
    confidence = (abs(probability_value - 0.5) * 2).round(3)

    return contributor_type, confidence


def __load_model(model_path=None):
    """
    Load a .joblib model from a given path.

    If no path is provided, the default model from RABBIT is loaded.

    Parameters:
        model_path (str) - path to the model to load (default: None)

    Returns:
        model - loaded model
    """
    if not model_path:
        model_path = files("rabbit").joinpath("resources", "models", "bimbas.joblib")

    try:
        bimbas_model = joblib.load(model_path)
        return bimbas_model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

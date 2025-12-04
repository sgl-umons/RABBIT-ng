import contextlib
import io
from importlib.resources import files

from ghmap.mapping.activity_mapper import ActivityMapper
from ghmap.mapping.action_mapper import ActionMapper
from ghmap.utils import load_json_file

from .features import compute_user_features
from .models import predict_contributor


def _compute_activity_sequences(events: list) -> list:
    """
    Compute activity sequences from the given events

    Args:
        events (list): List of event records
    Returns:
        list: List of activity sequences computed using ghmap
    """
    with contextlib.redirect_stdout(io.StringIO()):
        # Disable ghmap warnings in the stdout TODO: better way to handle ghmap logging?
        action_mapping_file = files("ghmap").joinpath("config", "event_to_action.json")
        action_mapping_json = load_json_file(action_mapping_file)
        action_mapper = ActionMapper(action_mapping_json, progress_bar=False)
        actions = action_mapper.map(events)

        activity_mapping_file = files("ghmap").joinpath(
            "config", "action_to_activity.json"
        )
        action_mapping_json = load_json_file(activity_mapping_file)
        activity_mapper = ActivityMapper(action_mapping_json, progress_bar=False)
        activities = activity_mapper.map(actions)

    return activities


def predict_user_type(username: str, events: list) -> tuple:
    """
    Predict the user type (bot or human) based on the given events
    """
    # TODO: manage return formats as in old rabbit
    activities = _compute_activity_sequences(events)
    if len(activities) == 0:
        # Events where found but no activities could be computed
        return "Unknown", "-"

    features_df = compute_user_features(username, activities)

    return predict_contributor(features_df)

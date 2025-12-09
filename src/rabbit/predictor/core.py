import contextlib
import io
from importlib.resources import files
import logging

from ghmap.mapping.activity_mapper import ActivityMapper
from ghmap.mapping.action_mapper import ActionMapper
from ghmap.utils import load_json_file

from .features import compute_user_features
from .models import Predictor

logger = logging.getLogger(__name__)


def _compute_activity_sequences(events: list) -> list:
    """
    Compute activity sequences from the given events

    Args:
        events (list): List of event records
    Returns:
        list: List of activity sequences computed using ghmap
    """
    # Suppress ghmap stdout output
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        action_mapping_file = files("ghmap").joinpath("config", "event_to_action.json")
        action_mapping_json = load_json_file(action_mapping_file)
        action_mapper = ActionMapper(action_mapping_json, progress_bar=False)
        actions = action_mapper.map(events)
        logger.debug(f"Mapped {len(events)} events to {len(actions)} actions.")

        activity_mapping_file = files("ghmap").joinpath(
            "config", "action_to_activity.json"
        )
        action_mapping_json = load_json_file(activity_mapping_file)
        activity_mapper = ActivityMapper(action_mapping_json, progress_bar=False)
        activities = activity_mapper.map(actions)
        logger.debug(f"Mapped {len(actions)} actions to {len(activities)} activities.")
    captured_output = stdout_capture.getvalue()
    if captured_output:
        # Filter output to keep only relevant debug info ("Warning: unused actions" and {actions})
        text = ""
        for line in captured_output.splitlines():
            if "Warning: Unused actions" in line or line.strip().startswith("{'"):
                # Keep only the part of the line after the warning
                line = line[line.index("Warning: Unused actions") :]
                text += line + "\n"

        if text:
            logger.debug("ghmap output: %s", text.strip())

    return activities


def predict_user_type(username: str, events: list, predictor: Predictor) -> tuple:
    """
    Predict the user type (bot or human) based on the given events
    """
    activities = _compute_activity_sequences(events)
    if len(activities) == 0:
        # Events where found but no activities could be computed
        logger.debug("No activity sequences found for user %s", username)
        return "Unknown", "-"

    features_df = compute_user_features(username, activities)

    return predictor.predict(features_df)

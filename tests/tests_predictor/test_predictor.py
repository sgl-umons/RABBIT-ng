from pathlib import Path

from rabbit.predictor import predict_user_type


class TestPredictor:
    def test_no_activities_returns_unknown(self):
        events = []

        username = "test"
        user_type, confidence = predict_user_type(username, events)
        assert user_type == "Unknown"
        assert confidence == "-"

    def test_predict_human(self):
        import json

        human_events_file = Path(__file__).parent.parent / "data" / "events_human.json"

        with open(human_events_file, "r", encoding="utf-8") as f:
            human_events = json.load(f)

        username = "test"
        user_type, confidence = predict_user_type(username, human_events)
        assert user_type == "Human"
        assert 0.0 <= confidence <= 1.0

from pathlib import Path

from rabbit.predictor import predict_user_type, ONNXPredictor


class TestPredictor:
    def test_no_activities_returns_unknown(self):
        events = []

        username = "test"
        result = predict_user_type(username, events, ONNXPredictor())
        assert result.user_type == "Unknown"
        assert result.confidence == "-"

    def test_predict_human(self):
        import json

        human_events_file = Path(__file__).parent.parent / "data" / "human_events.json"

        with open(human_events_file, "r", encoding="utf-8") as f:
            human_events = json.load(f)

        username = "test"
        result = predict_user_type(username, human_events, ONNXPredictor())

        assert result.user_type == "Human"
        assert 0.0 <= result.confidence <= 1.0

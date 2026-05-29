from pathlib import Path

from src.model.spike.tracking import MLflowTracker, NoopTracker, build_tracker


class FakeRunContext:
    def __init__(self, mlflow):
        self.mlflow = mlflow

    def __enter__(self):
        self.mlflow.events.append(("enter_run", None))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.mlflow.events.append(("exit_run", None))
        return False


class FakeMlflow:
    def __init__(self):
        self.events = []

    def set_tracking_uri(self, uri):
        self.events.append(("tracking_uri", uri))

    def set_experiment(self, name):
        self.events.append(("experiment", name))

    def start_run(self, run_name=None, nested=False):
        self.events.append(("start_run", run_name, nested))
        return FakeRunContext(self)

    def log_params(self, params):
        self.events.append(("params", params))

    def log_metrics(self, metrics):
        self.events.append(("metrics", metrics))

    def log_artifact(self, path):
        self.events.append(("artifact", str(path)))


def test_build_tracker_returns_noop_when_mlflow_disabled():
    tracker = build_tracker(enabled=False)

    assert isinstance(tracker, NoopTracker)


def test_mlflow_tracker_logs_params_metrics_and_artifacts(tmp_path):
    fake = FakeMlflow()
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nmacro_f1,0.5\n")

    tracker = MLflowTracker(
        mlflow_module=fake,
        experiment_name="spike_prediction",
        tracking_uri="file:./mlruns",
    )

    with tracker.start_run("run-a"):
        tracker.log_params({"target": "c_maize_fao", "horizon": 1})
        tracker.log_metrics({"macro_f1": 0.5})
        tracker.log_artifacts([artifact, tmp_path / "missing.csv"])

    assert ("tracking_uri", "file:./mlruns") in fake.events
    assert ("experiment", "spike_prediction") in fake.events
    assert ("params", {"target": "c_maize_fao", "horizon": 1}) in fake.events
    assert ("metrics", {"macro_f1": 0.5}) in fake.events
    assert ("artifact", str(artifact)) in fake.events
    assert not any(event[-1] == str(tmp_path / "missing.csv") for event in fake.events)

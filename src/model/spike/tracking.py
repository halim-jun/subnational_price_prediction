from contextlib import nullcontext
from pathlib import Path


class NoopTracker:
    def start_run(self, run_name: str | None = None, nested: bool = False):
        return nullcontext()

    def log_params(self, params: dict):
        return None

    def log_metrics(self, metrics: dict):
        return None

    def log_artifacts(self, paths):
        return None


class MLflowTracker:
    def __init__(
        self,
        mlflow_module,
        experiment_name: str,
        tracking_uri: str | None = None,
    ):
        self.mlflow = mlflow_module
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None, nested: bool = False):
        return self.mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: dict):
        self.mlflow.log_params(_clean_params(params))

    def log_metrics(self, metrics: dict):
        self.mlflow.log_metrics(_clean_metrics(metrics))

    def log_artifacts(self, paths):
        for path in paths:
            artifact_path = Path(path)
            if artifact_path.exists():
                self.mlflow.log_artifact(str(artifact_path))


def build_tracker(
    enabled: bool = True,
    experiment_name: str = "spike_prediction",
    tracking_uri: str = "file:./mlruns",
):
    if not enabled:
        return NoopTracker()
    try:
        import mlflow
    except ImportError:
        return NoopTracker()
    return MLflowTracker(
        mlflow_module=mlflow,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )


def _clean_params(params: dict) -> dict:
    return {key: _stringify_if_needed(value) for key, value in params.items()}


def _clean_metrics(metrics: dict) -> dict:
    clean = {}
    for key, value in metrics.items():
        try:
            metric = float(value)
        except (TypeError, ValueError):
            continue
        if metric == metric:
            clean[key] = metric
    return clean


def _stringify_if_needed(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

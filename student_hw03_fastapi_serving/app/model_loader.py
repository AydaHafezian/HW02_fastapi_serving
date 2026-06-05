from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """Load the HW02 model from MLflow."""

    def __init__(self) -> None:
        self.state = LoadedModelState()

    def _configure_mlflow_env(self) -> None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    def _resolve_run_id(self) -> Optional[str]:
        if config.MLFLOW_RUN_ID:
            return config.MLFLOW_RUN_ID

        client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1 DESC", "metrics.roc_auc DESC"],
            max_results=1,
        )
        if not runs:
            return None

        return runs[0].info.run_id

    def load(self) -> None:
        try:
            self._configure_mlflow_env()
            run_id = self._resolve_run_id()
            if not run_id:
                raise RuntimeError(
                    f"No MLflow run found for experiment '{config.MLFLOW_EXPERIMENT_NAME}'."
                )

            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
            run = client.get_run(run_id)

            self.state.model = model
            self.state.loaded = True
            self.state.error = None
            self.state.model_uri = model_uri
            self.state.run_id = run_id
            self.state.run_name = run.data.tags.get("mlflow.runName")
            self.state.metrics = dict(run.data.metrics)
            self.state.params = dict(run.data.params)
            self.state.tags = dict(run.data.tags)

        except Exception as exc:
            self.state.model = None
            self.state.loaded = False
            self.state.error = str(exc)

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
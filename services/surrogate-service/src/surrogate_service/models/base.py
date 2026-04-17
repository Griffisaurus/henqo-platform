"""
Base abstractions for surrogate models.

ModelRecord — immutable model metadata stored in the registry.
InferenceResult — output of model.predict().
SurrogateModel — abstract base class every surrogate must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelRecord:
    model_id: str
    surrogate_family: str
    weight_hash: str
    training_schema_version: str
    status: str                         # "production" | "staged" | "deprecated" | "frozen"
    training_dataset_revision: str
    n_ensemble_members: int = 8


@dataclass
class InferenceResult:
    outputs: dict[str, float]           # feature_name → predicted_value
    ensemble_outputs: list[float]       # raw ensemble member outputs (for A_ensemble)
    std_devs: dict[str, float]          # per-output std deviation
    model_id: str


class SurrogateModel(ABC):
    @abstractmethod
    def predict(self, x: dict[str, float]) -> InferenceResult:
        """Run inference and return an InferenceResult."""
        ...

    @abstractmethod
    def model_record(self) -> ModelRecord:
        """Return the immutable metadata record for this model."""
        ...

"""
In-memory model registry.

Stores SurrogateModel instances keyed by model_id and supports
production-model lookup by surrogate family.
"""
from __future__ import annotations

from surrogate_service.models.base import SurrogateModel


class ModelRegistry:
    """In-memory model registry for tests and local dev."""

    def __init__(self) -> None:
        self._by_id: dict[str, SurrogateModel] = {}

    def register(self, model: SurrogateModel) -> None:
        """Add or replace a model in the registry."""
        record = model.model_record()
        self._by_id[record.model_id] = model

    def get_production_model(self, surrogate_family: str) -> SurrogateModel | None:
        """
        Return the first model with status='production' for the given family.

        Returns None if no production model is found.
        """
        for model in self._by_id.values():
            rec = model.model_record()
            if rec.surrogate_family == surrogate_family and rec.status == "production":
                return model
        return None

    def get_by_id(self, model_id: str) -> SurrogateModel | None:
        """Return the model with the given model_id, or None."""
        return self._by_id.get(model_id)

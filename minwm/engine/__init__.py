"""minwm training and inference engines."""

from .events import EventStorage, get_event_storage, record_time
from .monitor import MetricsProcessor
from .recipe_base import Recipe
from .runtime import RuntimeContext, initialize_runtime
from .trainer import BaseTrainer

__all__ = [
    "Recipe",
    "BaseTrainer",
    "BaseInferencer",
    "RuntimeContext",
    "initialize_runtime",
    "EventStorage",
    "get_event_storage",
    "record_time",
    "MetricsProcessor",
]


def __getattr__(name: str):
    """Lazily expose inference without loading generation for training users."""
    if name == "BaseInferencer":
        from .inferencer import BaseInferencer

        return BaseInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

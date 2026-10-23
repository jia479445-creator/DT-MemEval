"""DT-MemEval's extracted MemBase API; optional runners load on first use."""
from importlib import import_module

_EXPORTS = {
    "MEMORY_LAYERS_MAPPING": ".layers", "CONFIG_MAPPING": ".configs",
    "DATASET_MAPPING": ".datasets", "ONLINE_EVAL_ENV_MAPPING": ".datasets",
    "METRIC_MAPPING": ".evaluation",
    "ConstructionRunnerConfig": ".runners.construction", "ConstructionRunner": ".runners.construction",
    "SearchRunnerConfig": ".runners.search", "SearchRunner": ".runners.search",
    "EvaluationRunnerConfig": ".runners.evaluation", "EvaluationRunner": ".runners.evaluation",
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

"""Load runtime/token-accounting dependencies only when required."""
from importlib import import_module
_EXPORTS = {
    "MonkeyPatcher": ".monkey_patch", "PatchSpec": ".monkey_patch", "make_attr_patch": ".monkey_patch",
    "token_monitor": ".token_monitor", "CostStateManager": ".token_monitor",
    "CostState": ".token_monitor", "get_tokenizer_for_model": ".token_monitor",
    "import_function_from_path": ".files", "download_models": ".files",
}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

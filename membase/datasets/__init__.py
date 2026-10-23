from collections import OrderedDict
from ..utils._lazy_mapping import _LazyMapping


_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        ("dualtimeline", "DualTimeline"),
    ]
)

_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
        ("dualtimeline", "dualtimeline"),
    ]
)

DATASET_MAPPING = _LazyMapping(
    mapping=_MAPPING_NAMES,
    module_mapping=_MODULE_MAPPING,
    package=__package__,
)

_ENV_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
    ]
)

_ENV_MODULE_MAPPING: OrderedDict[str, str] = OrderedDict(
    [
    ]
)

ONLINE_EVAL_ENV_MAPPING = _LazyMapping(
    mapping=_ENV_MAPPING_NAMES,
    module_mapping=_ENV_MODULE_MAPPING,
    package=__package__,
)

__all__ = ["DATASET_MAPPING", "ONLINE_EVAL_ENV_MAPPING"]

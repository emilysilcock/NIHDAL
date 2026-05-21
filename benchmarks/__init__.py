"""Benchmark dataset registry.

Each benchmark module registers a loader via the `@register(name)` decorator. Callers
go through `benchmarks.load(name, ...)`, which returns small_text TransformersDataset
objects for train and test (plus an optional list of bias indices if `biased=True`).

To add a new benchmark, drop a file in this directory that imports `register` and
decorates a loader function, then import the module here so the decorator runs.
"""

_REGISTRY = {}


def register(name):
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn

    return decorator


def load(name, tokenization_model, target_labels=(0,), biased=False, target_fraction=0.01):
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](
        tokenization_model=tokenization_model,
        target_labels=target_labels,
        biased=biased,
        target_fraction=target_fraction,
    )


def available():
    return sorted(_REGISTRY)


# Side-effect imports register the loaders with the registry.
from benchmarks import ag_news  # noqa: E402,F401
from benchmarks import trec  # noqa: E402,F401
from benchmarks import hate_speech  # noqa: E402,F401

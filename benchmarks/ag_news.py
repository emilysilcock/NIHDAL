import datasets

from benchmarks._transforms import format_binary_imbalanced
from benchmarks import register


@register("ag_news")
def load_ag_news(tokenization_model, target_labels=(0,), biased=False, target_fraction=0.01):
    raw = datasets.load_dataset("ag_news")
    return format_binary_imbalanced(
        raw,
        target_labels=list(target_labels),
        tokenization_model=tokenization_model,
        biased=biased,
        target_fraction=target_fraction,
    )

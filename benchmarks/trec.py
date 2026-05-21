import datasets

from benchmarks._transforms import format_binary_imbalanced
from benchmarks import register


@register("trec-10")
def load_trec10(tokenization_model, target_labels=(0,), biased=False, target_fraction=0.01):
    """TREC-10 coarse-grained question type classification (6 classes)."""
    raw = datasets.load_dataset("trec")

    # HF's `trec` exposes coarse labels as `coarse_label` and the text as `text`.
    raw = raw.rename_column("coarse_label", "label")

    return format_binary_imbalanced(
        raw,
        target_labels=list(target_labels),
        tokenization_model=tokenization_model,
        biased=biased,
        target_fraction=target_fraction,
    )

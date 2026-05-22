"""TREC-10 coarse-grained question type classification.

Loads the original TREC-10 (a.k.a. TREC question classification) dataset directly
from the canonical Cog Comp URLs, bypassing the HuggingFace Hub. The HF version
of `trec` shipped as a Python loader script, which `datasets>=3.0` no longer
supports.

Files (small — 600 KB combined):
    train: https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label   (5500 rows)
    test:  https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label      (500 rows)

Each line is `COARSE:FINE question text...` in Latin-1 encoding.
"""
import os
import urllib.request
from pathlib import Path

import datasets

from benchmarks._transforms import format_binary_imbalanced
from benchmarks import register


TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}

# 6 coarse labels, alphabetised so the index is reproducible across runs.
COARSE_LABELS = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]


def _cache_dir():
    """Cache the .label files under HF_HOME (if set) so they live alongside
    other HF datasets, otherwise fall back to ~/.cache/huggingface."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    d = Path(hf_home) / "datasets" / "trec_manual"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(split):
    path = _cache_dir() / f"{split}.label"
    if not path.exists():
        url = TREC_URLS[split]
        print(f"Downloading TREC {split} from {url}")
        urllib.request.urlretrieve(url, path)
    return path


def _parse(path):
    """Yield (text, coarse_label_index) tuples from a TREC .label file."""
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_part, _, text = line.partition(" ")
            coarse, _, _fine = label_part.partition(":")
            if coarse not in COARSE_LABELS:
                # Skip malformed line
                continue
            yield text, COARSE_LABELS.index(coarse)


def _load_split(split):
    rows = list(_parse(_fetch(split)))
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    return datasets.Dataset.from_dict(
        {"text": texts, "coarse_label": labels},
        features=datasets.Features({
            "text": datasets.Value("string"),
            "coarse_label": datasets.ClassLabel(names=COARSE_LABELS),
        }),
    )


@register("trec-10")
def load_trec10(tokenization_model, target_labels=(0,), biased=False, target_fraction=0.01):
    """TREC-10 coarse-grained question type classification (6 classes)."""
    raw = datasets.DatasetDict({
        "train": _load_split("train"),
        "test": _load_split("test"),
    })
    raw = raw.rename_column("coarse_label", "label")
    return format_binary_imbalanced(
        raw,
        target_labels=list(target_labels),
        tokenization_model=tokenization_model,
        biased=biased,
        target_fraction=target_fraction,
    )

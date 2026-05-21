import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks import register
from benchmarks._transforms import tokenize_to_transformers_dataset

RELIGIONS = [
    "atheist", "buddhist", "christian", "hindu", "jewish", "mormon", "muslim", "other",
]
SEED_SUBGROUP = "muslim"
SPLIT_RANDOM_STATE = 42
SPLIT_TEST_FRACTION = 0.2
HATE_SCORE_THRESHOLD = 1


def _aggregate_and_label():
    """Load the UCB measuring-hate-speech dataset, aggregate annotator rows by comment,
    and derive the binary label `(hate_speech_score > 1) & (target_religion == 1)`.
    Also computes a per-row `strat_col` used for the stratified train/test split and to
    identify the seed subgroup for biased runs (smallest-targeted-religion-first)."""
    raw = datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")
    dat = raw["train"].to_pandas()

    agg_cols = {"hate_speech_score": "mean", "target_religion": "mean"}
    for r in RELIGIONS:
        agg_cols[f"target_religion_{r}"] = "mean"
    dat = dat.groupby(["comment_id", "text"]).agg(agg_cols).reset_index()

    dat["label"] = ((dat["hate_speech_score"] > HATE_SCORE_THRESHOLD)
                    & (dat["target_religion"] == 1)).astype(int)

    strict_cols = []
    for r in RELIGIONS:
        col = f"target_religion_{r}_strict"
        dat[col] = (dat[f"target_religion_{r}"] == 1) & (dat["hate_speech_score"] > HATE_SCORE_THRESHOLD)
        strict_cols.append(col)

    targeted_count = dat[strict_cols].sum(axis=1).astype(int)
    inconsistent = (dat["label"] == 1) & (targeted_count == 0)
    if inconsistent.any():
        print(f"Dropping {int(inconsistent.sum())} rows with label=1 but no specific "
              "religion targeted")
        dat = dat[~inconsistent].reset_index(drop=True)

    counts = {r: int(dat[f"target_religion_{r}_strict"].sum()) for r in RELIGIONS}
    religions_by_size = sorted(RELIGIONS, key=lambda r: counts[r])

    dat["strat_col"] = "none"
    for r in religions_by_size:
        mask = (dat["strat_col"] == "none") & dat[f"target_religion_{r}_strict"]
        dat.loc[mask, "strat_col"] = r

    return dat[["text", "label", "strat_col"]]


def _stratified_split(df):
    train_parts, test_parts = [], []
    for strat_val, group in df.groupby("strat_col"):
        if len(group) < 2:
            train_parts.append(group)
            continue
        tr, te = train_test_split(
            group, test_size=SPLIT_TEST_FRACTION, random_state=SPLIT_RANDOM_STATE,
        )
        train_parts.append(tr)
        test_parts.append(te)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def _to_hf_dataset(df):
    features = datasets.Features({
        "text": datasets.Value("string"),
        "label": datasets.ClassLabel(names=["merged", "target"], num_classes=2),
    })
    return datasets.Dataset.from_pandas(df[["text", "label"]], features=features,
                                        preserve_index=False)


@register("hate_speech")
def load_hate_speech(tokenization_model, target_labels=(0,), biased=False,
                     target_fraction=0.01):
    """UC Berkeley measuring-hate-speech, binarised to detect religion-targeted hate
    speech (`hate_speech_score > 1 AND target_religion == 1`). The natural positive
    rate after annotator aggregation is ~1.35%, close enough to the AG News / TREC 1%
    target that no down-sampling is applied.

    `target_labels` and `target_fraction` are accepted for registry compatibility but
    ignored — the label is derived and the rate is left at its natural value.

    In biased mode the initial sample is restricted to Muslim-targeted positives (the
    largest religion subgroup); `bias_indices` lists positions of all *other* religion-
    targeted positives so `random_initialization_biased` can hold them out and the AL
    loop can track recovery.
    """
    df = _aggregate_and_label()
    train_df, test_df = _stratified_split(df)

    raw_dataset = {
        "train": _to_hf_dataset(train_df),
        "test": _to_hf_dataset(test_df),
    }
    train_dat, test_dat = tokenize_to_transformers_dataset(raw_dataset, tokenization_model)

    if biased:
        held_out_mask = (train_df["strat_col"] != SEED_SUBGROUP) & (train_df["label"] == 1)
        bias_indices = train_df.index[held_out_mask].tolist()
        return train_dat, test_dat, bias_indices
    return train_dat, test_dat

import datasets
import numpy as np
from small_text import TransformersDataset
from transformers import AutoTokenizer


def make_binary(dataset, target_labels):
    """Collapse a multi-class dataset to 2 classes: 1 for any label in `target_labels`,
    0 for everything else."""
    num_classes = dataset.features["label"].num_classes
    class_mapping = {lab: 0 for lab in range(num_classes)}
    for tl in target_labels:
        class_mapping[tl] = 1

    binary_dataset = dataset.map(lambda example: {"label": class_mapping[example["label"]]})

    new_features = datasets.Features({
        "text": binary_dataset.features["text"],
        "label": datasets.ClassLabel(names=["merged", "target"], num_classes=2),
    })
    return binary_dataset.cast(new_features)


def make_imbalanced(dataset, indices_to_track=None, target_fraction=0.01):
    """Down-sample the target class so it makes up `target_fraction` of the resulting
    dataset (default 1%). The non-target rows are kept intact.

    If `indices_to_track` is given (indices into the *pre-imbalance* dataset), the
    function additionally returns the post-imbalance positions of any of those points
    that survived the down-sampling. Used to keep track of a non-seeded sub-population
    of targets in the biased-initialization experiments.
    """
    other_samples = dataset.filter(lambda example: example["label"] == 0)
    target_samples = dataset.filter(lambda example: example["label"] == 1)

    other_samples_count = len(other_samples)
    imbalanced_total = other_samples_count / (1 - target_fraction)
    target_count = int(imbalanced_total * target_fraction)
    print(f"There are {target_count} target examples left in the dataset")

    target_samples = target_samples.shuffle()
    target_samples_to_keep = target_samples.select(range(target_count))

    imbalanced_dataset = datasets.concatenate_datasets([target_samples_to_keep, other_samples])

    if indices_to_track:
        target_list = [i for i in target_samples_to_keep]
        tracked_indices = []
        for idx in indices_to_track:
            point = dataset[idx]
            if point in target_list:
                tracked_indices.append(target_list.index(dataset[idx]))
        return imbalanced_dataset, tracked_indices

    return imbalanced_dataset


def tokenize_to_transformers_dataset(raw_dataset, tokenization_model, max_length=100):
    """Tokenize HF train/test datasets with two columns (text, label) into small_text
    TransformersDataset objects."""
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)
    num_classes = raw_dataset["train"].features["label"].num_classes
    lab_array = np.arange(num_classes)

    train_dat = TransformersDataset.from_arrays(
        raw_dataset["train"]["text"],
        raw_dataset["train"]["label"],
        tokenizer,
        max_length=max_length,
        target_labels=lab_array,
    )
    test_dat = TransformersDataset.from_arrays(
        raw_dataset["test"]["text"],
        raw_dataset["test"]["label"],
        tokenizer,
        max_length=max_length,
        target_labels=lab_array,
    )
    return train_dat, test_dat


def format_binary_imbalanced(raw_dataset, target_labels, tokenization_model, biased=False,
                              max_length=100, target_fraction=0.01):
    """Shared pipeline: binarize -> imbalance -> tokenize. Returns (train, test) or
    (train, test, bias_indices) if biased."""
    if biased:
        unsampled_train_indices = [
            i for i, lab in enumerate(raw_dataset["train"]["label"]) if lab == target_labels[1]
        ]

    raw_dataset["train"] = make_binary(raw_dataset["train"], target_labels)
    raw_dataset["test"] = make_binary(raw_dataset["test"], target_labels)

    if biased:
        raw_dataset["train"], bias_indices = make_imbalanced(
            raw_dataset["train"],
            indices_to_track=unsampled_train_indices,
            target_fraction=target_fraction,
        )
    else:
        raw_dataset["train"] = make_imbalanced(raw_dataset["train"], target_fraction=target_fraction)

    raw_dataset["test"] = make_imbalanced(raw_dataset["test"], target_fraction=target_fraction)

    train_dat, test_dat = tokenize_to_transformers_dataset(
        raw_dataset, tokenization_model, max_length=max_length
    )

    if biased:
        return train_dat, test_dat, bias_indices
    return train_dat, test_dat

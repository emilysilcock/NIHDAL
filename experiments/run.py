"""Run a single active-learning experiment.

Example:
    python -m experiments.run --dataset ag_news --method NIHDAL --seed 42 --biased
"""
import argparse
import logging
import pickle
import random
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import torch

from benchmarks import available as available_benchmarks
from benchmarks import load as load_benchmark
from experiments.learner import SUPPORTED_METHODS, set_up_active_learner
from experiments.loop import active_learning_loop

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=available_benchmarks(),
                        help="Benchmark name (see benchmarks/ for available loaders).")
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS,
                        help="Active learning method to run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--biased", action="store_true",
                        help="Use biased initialisation (seed half the target class only).")
    parser.add_argument("--transformer-model", default="distilroberta-base")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--query-batch-size", type=int, default=100)
    parser.add_argument("--initial-sample-size", type=int, default=100)
    parser.add_argument("--target-fraction", type=float, default=0.01,
                        help="Fraction of target class after imbalancing (default: 1%%).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-tag", default=None,
                        help="Optional suffix added to the result filename.")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Exit successfully if the output pickle already exists.")
    return parser.parse_args(argv)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def output_path(args):
    bias_tag = "biased" if args.biased else "unbiased"
    tag = f"_{args.output_tag}" if args.output_tag else ""
    method_slug = args.method.replace(" ", "_")
    return args.output_dir / f"{args.dataset}_{method_slug}_{args.seed}_{bias_tag}{tag}.pkl"


def main(argv=None):
    args = parse_args(argv)

    if args.skip_if_exists:
        out = output_path(args)
        if out.exists():
            print(f"Skipping: {out} already exists")
            return

    hf_datasets.logging.set_verbosity_error()
    hf_datasets.logging.get_verbosity = lambda: logging.NOTSET

    set_seed(args.seed)

    target_labels = [0, 1] if args.biased else [0]
    loaded = load_benchmark(
        args.dataset,
        tokenization_model=args.transformer_model,
        target_labels=target_labels,
        biased=args.biased,
        target_fraction=args.target_fraction,
    )
    if args.biased:
        train, test, bias_indices, bias_indices_test = loaded
    else:
        train, test = loaded
        bias_indices = None
        bias_indices_test = None

    active_learner = set_up_active_learner(
        args.transformer_model,
        active_learning_method=args.method,
        train=train,
        bias_indices=bias_indices,
    )

    results = active_learning_loop(
        active_learner,
        train,
        test,
        num_queries=args.num_queries,
        method=args.method,
        bias_indices=bias_indices,
        bias_indices_test=bias_indices_test,
        query_batch_size=args.query_batch_size,
        initial_sample_size=args.initial_sample_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = output_path(args)
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

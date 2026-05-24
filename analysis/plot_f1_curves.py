"""Plot Test F1 vs active-learning iteration, averaged across seeds.

Produces one figure per (dataset, biased|unbiased) combination, with all
methods overlaid. Reads the pickles written by `experiments.run` from
`--results-dir` and writes PNGs to `--output-dir`.

Example:
    python -m analysis.plot_f1_curves --results-dir results --output-dir results/figures
"""
import argparse
import json
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Kept in sync with benchmarks/__init__.py and experiments/learner.SUPPORTED_METHODS.
# Hardcoded here so this script imports nothing heavy (no torch / transformers).
DATASETS = ["ag_news", "trec-10", "hate_speech"]
METHODS = [
    "Random",
    "Least Confidence",
    "Prediction Entropy",
    "BALD",
    "BADGE",
    "Core Set",
    "Contrastive",
    "DAL",
    "NIHDAL",
    "NIHDAL_simon",
]

# Filename pattern: {dataset}_{method_slug}_{seed}_{bias}.pkl
# method_slug = method.replace(" ", "_") -- so spaces in method names become underscores.
FILENAME_RE = re.compile(r"^(?P<rest>.+)_(?P<seed>\d+)_(?P<bias>biased|unbiased)\.pkl$")


def parse_filename(name, datasets, method_slugs):
    """Return (dataset, method, seed, bias) or None if the name doesn't match."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    seed = int(m.group("seed"))
    bias = m.group("bias")
    for ds in datasets:
        prefix = ds + "_"
        if rest.startswith(prefix):
            slug = rest[len(prefix):]
            if slug in method_slugs:
                method = method_slugs[slug]
                return ds, method, seed, bias
    return None


def extract_f1_from_pickle(path):
    """Load one pickle and return its Test F1 trajectory.

    Used by the subprocess-per-file mode (see collect_runs) so the pickle's
    large embedding arrays are freed as soon as this process exits, keeping
    peak memory bounded to one file at a time.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    return [r["Test F1"] for r in res]


def collect_runs(results_dir, datasets, method_slugs, use_subprocess=True):
    """Return {(dataset, bias, method): {seed: [f1_per_iter]}}.

    With `use_subprocess=True`, each pickle is loaded in a child process.
    Pickles include large embedding arrays; loading them all in one process
    can OOM on a memory-capped login node.
    """
    runs = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(results_dir).glob("*.pkl")):
        parsed = parse_filename(p.name, datasets, method_slugs)
        if parsed is None:
            print(f"Skipping unparseable filename: {p.name}")
            continue
        ds, method, seed, bias = parsed
        if use_subprocess:
            r = subprocess.run(
                [sys.executable, __file__, "--extract-f1", str(p)],
                check=True, capture_output=True, text=True,
            )
            f1 = json.loads(r.stdout)
        else:
            f1 = extract_f1_from_pickle(p)
        runs[(ds, bias, method)][seed] = f1
    return runs


def aggregate(runs):
    """Average F1 across seeds. Returns {(ds, bias): {method: (iters, mean_f1)}}."""
    out = defaultdict(dict)
    for (ds, bias, method), per_seed in runs.items():
        arrs = list(per_seed.values())
        lens = {len(a) for a in arrs}
        if len(lens) > 1:
            n = min(lens)
            print(f"WARNING: {ds}/{bias}/{method} has seeds with different iter counts "
                  f"{sorted(lens)}; truncating to {n}")
            arrs = [a[:n] for a in arrs]
        mat = np.array(arrs)  # (n_seeds, n_iters)
        out[(ds, bias)][method] = (np.arange(mat.shape[1]), mat.mean(axis=0))
    return out


def plot(aggregated, output_dir, method_order):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (ds, bias), method_curves in sorted(aggregated.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = [m for m in method_order if m in method_curves]
        ordered += [m for m in method_curves if m not in method_order]
        for method in ordered:
            iters, f1 = method_curves[method]
            ax.plot(iters, f1, marker="o", label=method, linewidth=1.5, markersize=4)
        ax.set_xlabel("Active learning iteration")
        ax.set_ylabel("Test F1")
        ax.set_title(f"{ds} ({bias}) -- mean Test F1 across seeds")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = output_dir / f"f1_curve_{ds}_{bias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


def main():
    # Subprocess mode: load one pickle, print its F1 trajectory as JSON, exit.
    # Used internally by collect_runs to keep peak memory to one pickle at a time.
    if len(sys.argv) >= 3 and sys.argv[1] == "--extract-f1":
        print(json.dumps(extract_f1_from_pickle(sys.argv[2])))
        return

    ap = argparse.ArgumentParser(description=__doc__)
    default_results = Path(__file__).resolve().parents[1] / "results"
    ap.add_argument("--results-dir", type=Path, default=default_results)
    ap.add_argument("--output-dir", type=Path, default=default_results / "figures")
    ap.add_argument("--in-process", action="store_true",
                    help="Load all pickles in this process (faster, but uses lots of RAM).")
    args = ap.parse_args()

    method_slugs = {m.replace(" ", "_"): m for m in METHODS}

    runs = collect_runs(args.results_dir, DATASETS, method_slugs,
                        use_subprocess=not args.in_process)
    if not runs:
        raise SystemExit(f"No parseable pickles found in {args.results_dir}")

    aggregated = aggregate(runs)
    plot(aggregated, args.output_dir, method_order=METHODS)


if __name__ == "__main__":
    main()

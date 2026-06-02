"""Plot target-class examples selected per active-learning iteration.

For each (dataset, biased|unbiased) combination, plot one line per method
showing the mean (across seeds) number of selected examples that belong to
the target class at each AL iteration.

  - Unbiased runs: one panel per dataset, one line per method.
  - Biased runs: two panels per dataset side-by-side:
      left  -- target examples from the initialised target subset
      right -- target examples from the held-out (non-seeded) subset

Reads result pickles' `counts` field, written by `experiments.loop` per
iteration. Iteration 0 (the initial sample, not a query) has `counts=None`
and is skipped; the plotted x-axis is iteration 1..N.

Pickles include large embedding arrays, so each one is loaded in a
subprocess to keep peak memory bounded.

Example:
    python -m analysis.plot_target_selection --results-dir results --output-dir results/figures
"""
import argparse
import multiprocessing as mp
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

FILENAME_RE = re.compile(r"^(?P<rest>.+)_(?P<seed>\d+)_(?P<bias>biased|unbiased)\.pkl$")


def parse_filename(name, datasets, method_slugs):
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
                return ds, method_slugs[slug], seed, bias
    return None


def extract_counts(path):
    """Return list of per-iteration count dicts (one per query iter).

    Each dict has keys: 'target' (int), and -- for biased runs --
    'non_seeded_target' (int). Initial-sample iteration is skipped.
    """
    with open(path, "rb") as f:
        res = pickle.load(f)
    out = []
    for r in res:
        c = r.get("counts")
        if c is None:
            continue
        all_counts = c.get("all", {})
        rec = {"target": int(all_counts.get("target", 0))}
        if "non_seeded_target" in all_counts:
            rec["non_seeded_target"] = int(all_counts["non_seeded_target"])
        out.append(rec)
    return out


def _extract_one(args):
    """Worker entry point: load one pickle and return (key, counts_list)."""
    path_str, key = args
    return key, extract_counts(path_str)


def collect_runs(results_dir, datasets, method_slugs, processes=None, maxtasksperchild=20):
    """Return {(dataset, bias, method): {seed: [counts_dict_per_iter]}}.

    Pickles include large embedding arrays. multiprocessing.Pool with
    `maxtasksperchild` recycles workers after N tasks so memory stays
    bounded, while amortising the matplotlib/numpy import across pickles
    (the main reason this is ~10x faster than subprocess-per-file).
    """
    tasks = []
    for p in sorted(Path(results_dir).glob("*.pkl")):
        parsed = parse_filename(p.name, datasets, method_slugs)
        if parsed is None:
            print(f"Skipping unparseable filename: {p.name}")
            continue
        ds, method, seed, bias = parsed
        tasks.append((str(p), (ds, bias, method, seed)))

    runs = defaultdict(lambda: defaultdict(list))
    if not tasks:
        return runs

    processes = processes or min(8, mp.cpu_count())
    with mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for i, (key, counts) in enumerate(pool.imap_unordered(_extract_one, tasks, chunksize=1)):
            ds, bias, method, seed = key
            runs[(ds, bias, method)][seed] = counts
            if (i + 1) % 50 == 0 or i == len(tasks) - 1:
                print(f"Loaded {i + 1}/{len(tasks)} pickles", flush=True)
    return runs


def _mean_series(per_seed_series, key):
    """Mean across seeds of `[d[key] for d in series]`. Truncates to shortest series."""
    arrs = []
    for series in per_seed_series.values():
        arrs.append([d.get(key, 0) for d in series])
    if not arrs:
        return np.array([]), np.array([])
    n = min(len(a) for a in arrs)
    mat = np.array([a[:n] for a in arrs])
    iters = np.arange(1, n + 1)
    return iters, mat.mean(axis=0)


def plot_unbiased(method_curves, ax, title, method_order):
    ordered = [m for m in method_order if m in method_curves]
    ordered += [m for m in method_curves if m not in method_order]
    for method in ordered:
        iters, mean = _mean_series(method_curves[method], "target")
        ax.plot(iters, mean, marker="o", label=method, linewidth=1.5, markersize=4)
    ax.set_xlabel("Active learning iteration")
    ax.set_ylabel("Target examples selected (mean across seeds)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_biased(method_curves, axes, title_prefix, method_order):
    ax_init, ax_nons = axes
    ordered = [m for m in method_order if m in method_curves]
    ordered += [m for m in method_curves if m not in method_order]

    cmap = plt.cm.get_cmap("tab10", max(10, len(ordered)))
    colors = {m: cmap(i % cmap.N) for i, m in enumerate(ordered)}

    for method in ordered:
        per_seed = method_curves[method]
        iters_t, mean_t = _mean_series(per_seed, "target")
        iters_n, mean_n = _mean_series(per_seed, "non_seeded_target")
        mean_init = mean_t - mean_n  # initialised half
        ax_init.plot(iters_t, mean_init, marker="o", label=method,
                     color=colors[method], linewidth=1.5, markersize=4)
        ax_nons.plot(iters_n, mean_n, marker="o", label=method,
                     color=colors[method], linewidth=1.5, markersize=4)

    for ax, sub in [(ax_init, "target (initialised)"),
                    (ax_nons, "target (not initialised)")]:
        ax.set_xlabel("Active learning iteration")
        ax.set_ylabel(f"{sub} examples selected (mean across seeds)")
        ax.set_title(f"{title_prefix} -- {sub}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)


def plot(runs, output_dir, method_order):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_ds_bias = defaultdict(dict)
    for (ds, bias, method), per_seed in runs.items():
        by_ds_bias[(ds, bias)][method] = per_seed

    for (ds, bias), method_curves in sorted(by_ds_bias.items()):
        if bias == "unbiased":
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_unbiased(method_curves, ax,
                          title=f"{ds} (unbiased) -- target examples selected per iter",
                          method_order=method_order)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
            plot_biased(method_curves, axes,
                        title_prefix=f"{ds} (biased)",
                        method_order=method_order)
        fig.tight_layout()
        out = output_dir / f"target_selection_{ds}_{bias}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_results = Path(__file__).resolve().parents[1] / "results"
    ap.add_argument("--results-dir", type=Path, default=default_results)
    ap.add_argument("--output-dir", type=Path, default=default_results / "figures")
    ap.add_argument("--processes", type=int, default=None,
                    help="Pool worker count (default: min(8, cpu_count())).")
    args = ap.parse_args()

    method_slugs = {m.replace(" ", "_"): m for m in METHODS}

    runs = collect_runs(args.results_dir, DATASETS, method_slugs, processes=args.processes)
    if not runs:
        raise SystemExit(f"No parseable pickles found in {args.results_dir}")

    plot(runs, args.output_dir, method_order=METHODS)


if __name__ == "__main__":
    main()

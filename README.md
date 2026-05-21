# NIHDAL

Code for the NIHDAL active learning paper.

The paper's claim: standard active learning methods perform poorly when the target
class is extremely rare in the unlabelled pool, because their selection collapses
onto the majority class. **NIHDAL** (Negatively-Imbalanced Hierarchical Discriminative
Active Learning) restores balance by applying Discriminative Active Learning
separately to the model's predicted-target and predicted-other pools.

## Layout

```
nihdal/         # the method (NIHDAL, NIHDAL_2, PretrainedDiscriminativeActiveLearning)
benchmarks/     # benchmark dataset loaders (ag_news, trec-10); add new datasets here
experiments/    # CLI runner, AL loop, learner setup, init helpers, metrics
analysis/       # post-hoc analysis / plotting scripts
results/        # raw pickled results from experiments (gitignored)
archive/        # snapshot of earlier application-specific code (newspaper labelling)
```

## Install

```bash
uv sync
```

## Run an experiment

```bash
python -m experiments.run --dataset ag_news --method NIHDAL --seed 42 --biased
```

A run produces a pickle at `results/{dataset}_{method}_{seed}_{biased|unbiased}.pkl`
containing per-iteration metrics, embeddings, and selection diagnostics.

Available methods (see `experiments/learner.py`):
Random, Least Confidence, Prediction Entropy, BALD, BADGE, Core Set, Contrastive,
DAL, NIHDAL.

Available benchmarks (see `benchmarks/`): `ag_news`, `trec-10`. New benchmarks: drop
a loader file into `benchmarks/`, decorate with `@register("name")`, and import it
from `benchmarks/__init__.py`.

## Sweeps

Run the full grid with a shell loop, e.g.:

```bash
for ds in ag_news trec-10 hate_speech; do
  for m in Random "Least Confidence" "Prediction Entropy" BALD BADGE "Core Set" Contrastive DAL NIHDAL; do
    for seed in 42 12731 65372; do
      for bias in --biased ""; do
        python -m experiments.run --dataset $ds --method "$m" --seed $seed $bias
      done
    done
  done
done
```

import numpy as np

from small_text import DiscriminativeActiveLearning

from nihdal.pretrained_dal import PretrainedDiscriminativeActiveLearning


def _coerce_bias_indices(bias_indices):
    if bias_indices is None:
        return None
    return frozenset(int(i) for i in bias_indices)


def _pool_summary(indices, dataset, bias_indices):
    """Per-pool diagnostic record: how many were picked, how many are actually target,
    and (optional) how many came from the non-seeded subset."""
    indices = np.asarray(indices, dtype=np.int64)
    summary = {
        "selected": int(len(indices)),
        "target": int(dataset.y[indices].sum()) if len(indices) else 0,
    }
    if bias_indices is not None:
        summary["non_seeded_target"] = int(
            sum(1 for i in indices if int(i) in bias_indices)
        )
    return summary


def _log_pool_stats(label, indices, dataset, bias_indices):
    """Print a one-line stats summary for a pool. Errors are reported, not swallowed."""
    if len(indices) == 0:
        print(f"There are 0 {label} examples in the unlabeled pool")
        return
    try:
        target_count = int(dataset.y[indices].sum())
    except Exception as exc:
        print(f"WARNING: failed to count targets in {label} pool: "
              f"{type(exc).__name__}: {exc}")
        return
    msg = (f"There are {len(indices)} {label} examples in the unlabeled pool, "
           f"of which {target_count} are actually target")
    if bias_indices is not None:
        bias_count = sum(1 for i in indices if int(i) in bias_indices)
        msg += f" ({bias_count} in the non-seeded target subset)"
    print(msg)


def _select_from_pool(strategy, clf, dataset, indices_unlabeled, indices_labeled, k):
    """Pick `k` indices from `indices_unlabeled` via DAL, with fallbacks:

      - k <= 0                          → empty
      - len(indices_unlabeled) <= k     → take the whole pool
      - len(indices_labeled) == 0       → DAL can't train; random-sample instead

    Otherwise: run DAL via `strategy._discriminative_active_learning`.
    """
    indices_unlabeled = np.asarray(indices_unlabeled, dtype=np.int64)
    indices_labeled = np.asarray(indices_labeled, dtype=np.int64)

    if k <= 0:
        return np.array([], dtype=np.int64)
    if len(indices_unlabeled) <= k:
        return indices_unlabeled
    if len(indices_labeled) == 0:
        print(f"  No labeled examples on this side; random-selecting {k}")
        return np.random.choice(indices_unlabeled, k, replace=False).astype(np.int64)

    query_sizes = DiscriminativeActiveLearning._get_query_sizes(strategy.num_iterations, k)
    return np.asarray(
        strategy._discriminative_active_learning(
            clf, dataset, indices_unlabeled, indices_labeled, query_sizes,
        ),
        dtype=np.int64,
    )


class NIHDAL(PretrainedDiscriminativeActiveLearning):
    """NIHDAL: applies Discriminative Active Learning separately to the predicted-target
    and predicted-other pools, so that selection does not collapse onto the majority
    class when the target class is very rare.

    `bias_indices` is an optional collection of indices (the "non-seeded" target subset
    used in biased-initialization experiments) used for diagnostic logging only.
    """

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10,
                 pbar="tqdm", bias_indices=None):
        super().__init__(classifier_factory, num_iterations, unlabeled_factor, pbar)
        self.bias_indices = _coerce_bias_indices(bias_indices)
        self.last_selected_descr = {}

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        indices_unlabeled = np.asarray(indices_unlabeled, dtype=np.int64)
        indices_labeled = np.asarray(indices_labeled, dtype=np.int64)

        preds = np.asarray(clf.predict(dataset))
        labels = np.asarray(dataset.y)

        target_indices_unlabeled = indices_unlabeled[preds[indices_unlabeled] == 1]
        other_indices_unlabeled = indices_unlabeled[preds[indices_unlabeled] == 0]
        target_indices_labeled = indices_labeled[labels[indices_labeled] == 1]
        other_indices_labeled = indices_labeled[labels[indices_labeled] == 0]

        _log_pool_stats("predicted target", target_indices_unlabeled, dataset, self.bias_indices)
        _log_pool_stats("predicted other", other_indices_unlabeled, dataset, self.bias_indices)

        half = n // 2

        if len(target_indices_unlabeled) <= half:
            print("Few predicted targets — taking all and filling with DAL on other pool")
            target_indices = target_indices_unlabeled
            other_indices = _select_from_pool(
                self, clf, dataset, other_indices_unlabeled, other_indices_labeled,
                n - len(target_indices),
            )
        elif len(other_indices_unlabeled) <= half:
            print("Few predicted non-targets — taking all and filling with DAL on target pool")
            other_indices = other_indices_unlabeled
            target_indices = _select_from_pool(
                self, clf, dataset, target_indices_unlabeled, target_indices_labeled,
                n - len(other_indices),
            )
        else:
            target_indices = _select_from_pool(
                self, clf, dataset, target_indices_unlabeled, target_indices_labeled, half,
            )
            other_indices = _select_from_pool(
                self, clf, dataset, other_indices_unlabeled, other_indices_labeled, n - half,
            )

        selected_indices = np.concatenate((target_indices, other_indices)).astype(np.int64)

        self.last_selected_descr = {
            "predicted_target": _pool_summary(target_indices, dataset, self.bias_indices),
            "predicted_other": _pool_summary(other_indices, dataset, self.bias_indices),
            "all": _pool_summary(selected_indices, dataset, self.bias_indices),
        }
        return selected_indices


class NIHDAL_2(PretrainedDiscriminativeActiveLearning):
    """NIHDAL variant: instead of running DAL on the two pools separately, build a
    class-balanced pool from the model's predicted targets and predicted others,
    then run DAL once on that balanced pool.
    """

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10,
                 pbar="tqdm", bias_indices=None):
        super().__init__(classifier_factory, num_iterations, unlabeled_factor, pbar)
        self.bias_indices = _coerce_bias_indices(bias_indices)
        self.last_selected_descr = {}

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        indices_unlabeled = np.asarray(indices_unlabeled, dtype=np.int64)
        indices_labeled = np.asarray(indices_labeled, dtype=np.int64)

        preds = np.asarray(clf.predict(dataset))

        target_indices_unlabeled = indices_unlabeled[preds[indices_unlabeled] == 1]
        other_indices_unlabeled = indices_unlabeled[preds[indices_unlabeled] == 0]

        _log_pool_stats("predicted target", target_indices_unlabeled, dataset, self.bias_indices)
        _log_pool_stats("predicted other", other_indices_unlabeled, dataset, self.bias_indices)

        # If either side is empty, fall back to running DAL on whatever is left.
        if len(target_indices_unlabeled) == 0 or len(other_indices_unlabeled) == 0:
            print("One side empty — falling back to DAL on the full unlabeled pool")
            selected_indices = _select_from_pool(
                self, clf, dataset, indices_unlabeled, indices_labeled, n,
            )
        else:
            half_pool_size = min(len(target_indices_unlabeled), len(other_indices_unlabeled))
            target_pool = np.random.choice(target_indices_unlabeled, half_pool_size, replace=False)
            other_pool = np.random.choice(other_indices_unlabeled, half_pool_size, replace=False)
            balanced_indices_unlabeled = np.concatenate((target_pool, other_pool)).astype(np.int64)

            selected_indices = _select_from_pool(
                self, clf, dataset, balanced_indices_unlabeled, indices_labeled, n,
            )

        selected_indices = np.asarray(selected_indices, dtype=np.int64)

        # Attribute each selected index to its pool of origin for diagnostics.
        target_set = set(int(i) for i in target_indices_unlabeled)
        from_target = np.array(
            [i for i in selected_indices if int(i) in target_set], dtype=np.int64,
        )
        from_other = np.array(
            [i for i in selected_indices if int(i) not in target_set], dtype=np.int64,
        )

        self.last_selected_descr = {
            "predicted_target": _pool_summary(from_target, dataset, self.bias_indices),
            "predicted_other": _pool_summary(from_other, dataset, self.bias_indices),
            "all": _pool_summary(selected_indices, dataset, self.bias_indices),
        }
        return selected_indices

import numpy as np

from experiments.evaluate import evaluate
from experiments.initialization import (
    initialize_active_learner_balanced,
    initialize_active_learner_biased,
)

NIHDAL_METHODS = {"NIHDAL", "NIHDAL_simon"}


def active_learning_loop(active_learner, train, test, num_queries, method, bias_indices=None,
                         query_batch_size=100, initial_sample_size=100):
    """Run the AL loop. `method` is the human-readable method name (e.g. 'NIHDAL').
    `bias_indices` is None for unbiased runs, or a list of held-out target indices for
    biased-init runs (used both to skip during initialisation and for diagnostics).
    """
    if bias_indices is not None:
        indices_labeled = initialize_active_learner_biased(
            active_learner, train.y, bias_indices, n_samples=initial_sample_size
        )
    else:
        indices_labeled = initialize_active_learner_balanced(
            active_learner, train.y, n_samples=initial_sample_size
        )

    print(f"Initial sample contains {sum(train.y[indices_labeled])} target class")
    if bias_indices is not None:
        in_bias = [i for i in indices_labeled if i in bias_indices]
        print(f"Initial sample contains {len(in_bias)} from non-seeded target class")

    results = [evaluate(active_learner, train[indices_labeled], test)]

    for i in range(num_queries):
        indices_queried = active_learner.query(num_samples=query_batch_size)
        y = train.y[indices_queried]
        active_learner.update(y)
        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print("---------------")
        print(f"Iteration #{i} ({len(indices_labeled)} samples)")
        res = evaluate(active_learner, train[indices_labeled], test)

        if method in NIHDAL_METHODS:
            selected_descr = getattr(active_learner.query_strategy, "last_selected_descr", {})
        else:
            selected_descr = {
                "all": {
                    "selected": len(indices_queried),
                    "target": int(sum(y)),
                }
            }
            if bias_indices is not None:
                selected_descr["all"]["non_seeded_target"] = len(
                    [i for i in indices_queried if i in bias_indices]
                )

        res["counts"] = selected_descr
        print(selected_descr)
        results.append(res)

    return results

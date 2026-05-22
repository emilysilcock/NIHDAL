import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, pairwise_distances,
    pairwise_distances_argmin_min, precision_score, recall_score,
)


def _min_within(embeddings):
    """Mean and max of min-distance within a set (excluding self-pairs)."""
    d = pairwise_distances(embeddings)
    np.fill_diagonal(d, np.inf)
    min_d = d.min(axis=1)
    return float(min_d.mean()), float(min_d.max())


def evaluate(active_learner, train, indices_labeled, test, bias_indices_test=None):
    indices_labeled = np.asarray(indices_labeled, dtype=np.int64)
    train_labelled = train[indices_labeled]

    y_pred_labelled = active_learner.classifier.predict(train_labelled)
    y_pred_test = active_learner.classifier.predict(test)

    labelled_embeddings = active_learner.classifier.embed(train_labelled)
    test_embeddings = active_learner.classifier.embed(test)
    train_embeddings = active_learner.classifier.embed(train)

    r = {
        "Train accuracy": accuracy_score(train_labelled.y, y_pred_labelled),
        "Test accuracy": accuracy_score(test.y, y_pred_test),
        "Train F1": f1_score(train_labelled.y, y_pred_labelled),
        "Test F1": f1_score(test.y, y_pred_test),
        "Train precision": precision_score(train_labelled.y, y_pred_labelled),
        "Test precision": precision_score(test.y, y_pred_test),
        "Train recall": recall_score(train_labelled.y, y_pred_labelled),
        "Test recall": recall_score(test.y, y_pred_test),
        "Test predictions": y_pred_test,
        "Test ground truth": test.y,
        "Test embeddings": test_embeddings,
        "Train embeddings": train_embeddings,
        "Labelled data embeddings": labelled_embeddings,
        "Labelled data labels": train_labelled.y,
    }

    if bias_indices_test is not None and len(bias_indices_test) > 0:
        ns = np.asarray(bias_indices_test, dtype=np.int64)
        y_true_ns = np.asarray(test.y)[ns]
        y_pred_ns = np.asarray(y_pred_test)[ns]
        r["Test non_seeded accuracy"] = accuracy_score(y_true_ns, y_pred_ns)
        r["Test non_seeded F1"] = f1_score(y_true_ns, y_pred_ns, zero_division=0)
        r["Test non_seeded precision"] = precision_score(y_true_ns, y_pred_ns, zero_division=0)
        r["Test non_seeded recall"] = recall_score(y_true_ns, y_pred_ns, zero_division=0)

    if len(labelled_embeddings) >= 2:
        mean_d, max_d = _min_within(labelled_embeddings)
        r["Labelled diversity mean_min_dist"] = mean_d
        r["Labelled diversity max_min_dist"] = max_d

        pos_mask = np.asarray(train_labelled.y) == 1
        if pos_mask.sum() >= 2:
            mean_dp, max_dp = _min_within(labelled_embeddings[pos_mask])
            r["Labelled diversity (positive) mean_min_dist"] = mean_dp
            r["Labelled diversity (positive) max_min_dist"] = max_dp

    unlabelled_mask = np.ones(len(train), dtype=bool)
    unlabelled_mask[indices_labeled] = False
    if unlabelled_mask.any():
        unlab_embeddings = train_embeddings[unlabelled_mask]
        _, min_d = pairwise_distances_argmin_min(unlab_embeddings, labelled_embeddings)
        r["Train coverage mean_min_dist"] = float(min_d.mean())
        r["Train coverage max_min_dist"] = float(min_d.max())

        unlab_y = np.asarray(train.y)[unlabelled_mask]
        pos_unlab = unlab_y == 1
        if pos_unlab.any():
            r["Train coverage (positive) mean_min_dist"] = float(min_d[pos_unlab].mean())
            r["Train coverage (positive) max_min_dist"] = float(min_d[pos_unlab].max())

    print("Test accuracy:", r["Test accuracy"], "Test F1:", r["Test F1"])
    if "Test non_seeded F1" in r:
        print("Test non-seeded F1:", r["Test non_seeded F1"],
              "recall:", r["Test non_seeded recall"])
    return r

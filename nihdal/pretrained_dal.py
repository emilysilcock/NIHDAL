import gc

import numpy as np
import torch

from small_text import DiscriminativeActiveLearning
from small_text.query_strategies import QueryStrategy
from small_text.utils.context import build_pbar_context


class PretrainedDiscriminativeActiveLearning(QueryStrategy):
    """Discriminative Active Learning that initializes the discriminative transformer
    classifier with weights from the current task classifier.
    """

    LABEL_LABELED_POOL = 0
    LABEL_UNLABELED_POOL = 1

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10, pbar="tqdm"):
        self.classifier_factory = classifier_factory
        self.num_iterations = num_iterations
        self.unlabeled_factor = unlabeled_factor
        self.pbar = pbar
        self.clf_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        query_sizes = DiscriminativeActiveLearning._get_query_sizes(self.num_iterations, n)
        return self._discriminative_active_learning(
            clf, dataset, indices_unlabeled, indices_labeled, query_sizes
        )

    def _discriminative_active_learning(self, clf, dataset, indices_unlabeled, indices_labeled, query_sizes):
        indices = np.array([], dtype=indices_labeled.dtype)
        indices_unlabeled_copy = np.copy(indices_unlabeled)
        indices_labeled_copy = np.copy(indices_labeled)

        with build_pbar_context(len(query_sizes)) as pbar:
            for q in query_sizes:
                indices_most_confident = self._train_and_get_most_confident(
                    clf, dataset, indices_unlabeled_copy, indices_labeled_copy, q
                )

                indices = np.append(indices, indices_unlabeled_copy[indices_most_confident])
                indices_labeled_copy = np.append(
                    indices_labeled_copy, indices_unlabeled_copy[indices_most_confident]
                )
                indices_unlabeled_copy = np.delete(indices_unlabeled_copy, indices_most_confident)
                pbar.update(1)

        return indices

    def _train_and_get_most_confident(self, clf, ds, indices_unlabeled, indices_labeled, q):
        if self.clf_ is not None:
            del self.clf_
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        original_num_classes = self.classifier_factory.num_classes
        self.classifier_factory.num_classes = 2
        discr_clf = self.classifier_factory.new()
        self.classifier_factory.num_classes = original_num_classes

        num_unlabeled = min(
            indices_labeled.shape[0] * self.unlabeled_factor, indices_unlabeled.shape[0]
        )
        indices_unlabeled_sub = np.random.choice(indices_unlabeled, num_unlabeled, replace=False)

        ds_discr = DiscriminativeActiveLearning.get_relabeled_copy(
            ds, indices_unlabeled_sub, indices_labeled
        )

        if hasattr(discr_clf, "initialize"):
            discr_clf.initialize()

        # Warm-start the discriminative model with the task classifier's encoder weights.
        if (
            hasattr(clf, "model")
            and hasattr(discr_clf, "model")
            and clf.model is not None
            and discr_clf.model is not None
        ):
            try:
                main_state_dict = clf.model.state_dict()
                discr_state_dict = discr_clf.model.state_dict()

                for name, param in main_state_dict.items():
                    if (
                        "classifier" not in name
                        and "head" not in name
                        and name in discr_state_dict
                        and discr_state_dict[name].shape == param.shape
                    ):
                        discr_state_dict[name].copy_(param)

                discr_clf.model.load_state_dict(discr_state_dict)
                self.clf_ = discr_clf.fit(ds_discr)
            except Exception as e:
                discr_clf.fit(ds_discr)
                self.clf_ = discr_clf
                print(f"Weight transfer failed, training from scratch. Error: {e}")
        else:
            print("No weights to transfer, training from scratch")
            discr_clf.fit(ds_discr)
            self.clf_ = discr_clf

        proba = self.clf_.predict_proba(ds[indices_unlabeled])
        proba = proba[:, self.LABEL_UNLABELED_POOL]

        return np.argpartition(-proba, q)[:q]

    def __str__(self):
        return (
            f"PretrainedDiscriminativeActiveLearning("
            f"classifier_factory={str(self.classifier_factory)}, "
            f"num_iterations={self.num_iterations}, "
            f"unlabeled_factor={self.unlabeled_factor})"
        )

import numpy as np

from small_text import DiscriminativeActiveLearning

from nihdal.pretrained_dal import PretrainedDiscriminativeActiveLearning


class NIHDAL(PretrainedDiscriminativeActiveLearning):
    """NIHDAL: applies Discriminative Active Learning separately to the predicted-target
    and predicted-other pools, so that selection does not collapse onto the majority class
    when the target class is very rare.

    `bias_indices` is an optional set of indices (the "non-seeded" target subset used in
    biased-initialization experiments) used only for diagnostic logging.
    """

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10,
                 pbar="tqdm", bias_indices=None):
        super().__init__(classifier_factory, num_iterations, unlabeled_factor, pbar)
        self.bias_indices = bias_indices
        self.last_selected_descr = {}

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        preds = clf.predict(dataset)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        target_indices_labeled = np.array([i for i in indices_labeled if dataset.y[i] == 1])
        other_indices_labeled = np.array([i for i in indices_labeled if dataset.y[i] == 0])

        self._log_pool_stats("target", target_indices_unlabeled, dataset)
        self._log_pool_stats("non-target", other_indices_unlabeled, dataset)

        if len(target_indices_unlabeled) <= n / 2:
            print("Classification model predicted few targets")
            target_indices = np.array(target_indices_unlabeled)
            query_sizes = DiscriminativeActiveLearning._get_query_sizes(
                self.num_iterations, n - len(target_indices)
            )
            print("Finding others to label ...")
            other_indices = self._discriminative_active_learning(
                clf, dataset, other_indices_unlabeled, other_indices_labeled, query_sizes
            )
        elif len(other_indices_unlabeled) <= n / 2:
            print("Classification model predicted few non-targets, reverting to DAL")
            other_indices = np.array(other_indices_unlabeled)
            query_sizes = DiscriminativeActiveLearning._get_query_sizes(
                self.num_iterations, n - len(other_indices)
            )
            print("Finding targets to label ...")
            target_indices = self._discriminative_active_learning(
                clf, dataset, target_indices_unlabeled, target_indices_labeled, query_sizes
            )
        else:
            query_sizes = DiscriminativeActiveLearning._get_query_sizes(
                self.num_iterations, int(n / 2)
            )
            print("Finding targets to label ...")
            target_indices = self._discriminative_active_learning(
                clf, dataset, target_indices_unlabeled, target_indices_labeled, query_sizes
            )
            print("Finding others to label ...")
            other_indices = self._discriminative_active_learning(
                clf, dataset, other_indices_unlabeled, other_indices_labeled, query_sizes
            )

        selected_indices = np.concatenate((target_indices, other_indices)).astype(int)
        self.last_selected_descr = {}
        return selected_indices

    def _log_pool_stats(self, label, indices, dataset):
        try:
            target_count = sum(dataset.y[indices])
            print(
                f"There are {len(indices)} predicted {label} examples, "
                f"of which {target_count} are actually target"
            )
            if self.bias_indices is not None:
                bias_count = len([i for i in self.bias_indices if i in indices])
                print(f"of these {bias_count} are in the non-seeded target")
        except Exception:
            pass


class NIHDAL_2(PretrainedDiscriminativeActiveLearning):
    """NIHDAL variant: instead of running DAL on the two pools separately, build a
    class-balanced pool from the model's predicted targets and predicted others,
    then run DAL once on that balanced pool.
    """

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10,
                 pbar="tqdm", bias_indices=None):
        super().__init__(classifier_factory, num_iterations, unlabeled_factor, pbar)
        self.bias_indices = bias_indices
        self.last_selected_descr = {}

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        preds = clf.predict(dataset)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        target_count = sum(dataset.y[target_indices_unlabeled])
        print(
            f"There are {len(target_indices_unlabeled)} predicted target examples, "
            f"of which {target_count} are actually target"
        )
        if self.bias_indices is not None:
            bias_count = len([i for i in self.bias_indices if i in target_indices_unlabeled])
            print(f"of these {bias_count} are in the non-seeded target")

        other_count = sum(dataset.y[other_indices_unlabeled])
        print(
            f"There are {len(other_indices_unlabeled)} predicted non-target examples, "
            f"of which {other_count} are actually target"
        )
        if self.bias_indices is not None:
            bias_count = len([i for i in self.bias_indices if i in other_indices_unlabeled])
            print(f"of these {bias_count} are in the non-seeded target")

        half_pool_size = min(len(target_indices_unlabeled), len(other_indices_unlabeled))
        target_pool = np.random.choice(target_indices_unlabeled, half_pool_size, replace=False)
        other_pool = np.random.choice(other_indices_unlabeled, half_pool_size, replace=False)
        balanced_indices_unlabeled = np.concatenate((target_pool, other_pool)).astype(int)

        self._validate_query_input(balanced_indices_unlabeled, n)
        query_sizes = DiscriminativeActiveLearning._get_query_sizes(self.num_iterations, int(n))

        selected_indices = self._discriminative_active_learning(
            clf, dataset, balanced_indices_unlabeled, indices_labeled, query_sizes
        )

        self.last_selected_descr = {}
        return selected_indices

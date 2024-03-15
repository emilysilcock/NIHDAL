import numpy as np
from small_text import DiscriminativeActiveLearning


class DiscriminativeActiveLearning_amended(DiscriminativeActiveLearning):

    # Amended to use the most recent topic classifier as per the DAL paper

    def _train_and_get_most_confident(self, ds, indices_unlabeled, indices_labeled, q):

        if self.clf_ is not None:
            del self.clf_

        clf = active_learner._clf
        # clf = deepcopy(last_stable_model)

        num_unlabeled = min(indices_labeled.shape[0] * self.unlabeled_factor,
                            indices_unlabeled.shape[0])

        indices_unlabeled_sub = np.random.choice(indices_unlabeled,
                                                num_unlabeled,
                                                replace=False)

        ds_discr = DiscriminativeActiveLearning_amended.get_relabeled_copy(ds,
                                                                indices_unlabeled_sub,
                                                                indices_labeled)

        self.clf_ = clf.fit(ds_discr)

        proba = clf.predict_proba(ds[indices_unlabeled])
        proba = proba[:, self.LABEL_UNLABELED_POOL]

        # return instances which most likely belong to the "unlabeled" class (higher is better)
        return np.argpartition(-proba, q)[:q]


class NIHDAL(DiscriminativeActiveLearning_amended):

    """Similar to Discriminative Active Learning, but applied on the predicted target and 
     others separately. 
    """

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):

        self._validate_query_input(indices_unlabeled, n)

        # Predict target or other for unlabelled data
        # preds = last_stable_model.predict(train)
        preds = active_learner.classifier.predict(train)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        target_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 1])
        other_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 0])

        # Describe predicted target
        target_indices_unlabeled_tar_count = sum(train.y[target_indices_unlabeled])
        print(f'There are {len(target_indices_unlabeled)} predicted target examples, of which {target_indices_unlabeled_tar_count} are actually target')
        if biased:
            bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in target_indices_unlabeled])
            print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')

        # Describe predicted other
        other_indices_unlabeled_tar_count = sum(train.y[other_indices_unlabeled])
        print(f'There are {len(other_indices_unlabeled)} predicted non-target examples, of which {other_indices_unlabeled_tar_count} are actually target')
        if biased:
            bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in other_indices_unlabeled])
            print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')

        # If there are not enough predicted targets
        if len(target_indices_unlabeled) < n/2:
            print("Classification model predicted few targets")

            # Take all predicted targets
            target_indices = np.array(target_indices_unlabeled)

            # Run normal DAL for the rest
            query_sizes = self._get_query_sizes(self.num_iterations, n - len(target_indices))

            print("Finding others to label ...")
            other_indices = self.discriminative_active_learning(
                dataset,
                other_indices_unlabeled,
                other_indices_labeled,
                query_sizes
            )

        # If there are not enough predicted others
        elif len(other_indices_unlabeled) < n/2:

            print("Classification model predicted few non-targets, reverting to DAL")

            # Take all other targets
            other_indices = np.array(target_indices_unlabeled)

            # Run normal DAL for the rest
            query_sizes = self._get_query_sizes(self.num_iterations, n - len(other_indices))

            print("Finding targets to label ...")
            target_indices = self.discriminative_active_learning(
                dataset,
                target_indices_unlabeled,
                target_indices_labeled,
                query_sizes
            )

        else:
            query_sizes = self._get_query_sizes(self.num_iterations, int(n/2))

            print("Finding targets to label ...")
            target_indices = self.discriminative_active_learning(
                dataset,
                target_indices_unlabeled,
                target_indices_labeled,
                query_sizes
            )
            print("Finding others to label ...")
            other_indices = self.discriminative_active_learning(
                dataset,
                other_indices_unlabeled,
                other_indices_labeled,
                query_sizes
            )

        selected_indices = np.concatenate((target_indices, other_indices)).astype(int)

        # Describe selected indices
        ## From pred target

        global selected_descr
        selected_descr = {}

        pred_tar_actual_tar = sum(train.y[target_indices])
        print(f'Predicted target: Selected {len(target_indices)} samples, with {pred_tar_actual_tar} target class')
        selected_descr['predicted_target'] = {}
        selected_descr['predicted_target']['selected'] = len(target_indices)
        selected_descr['predicted_target']['target'] = pred_tar_actual_tar
        if biased:
            bias_indices_selected_tar = len([i for i in target_indices if i in bias_indices])
            print(f'of these {bias_indices_selected_tar} are in the non-seeded target')
            selected_descr['predicted_target']['non_seeded_target'] = bias_indices_selected_tar

        ## From pred other
        pred_oth_actual_tar = sum(train.y[other_indices])
        print(f'Predicted non-target: Selected {len(other_indices)} samples, with {pred_oth_actual_tar} target class')
        selected_descr['predicted_other'] = {}
        selected_descr['predicted_other']['selected'] = len(other_indices)
        selected_descr['predicted_other']['target'] = pred_oth_actual_tar
        if biased:
            bias_indices_selected_oth = len([i for i in other_indices if i in bias_indices])
            print(f'of these {bias_indices_selected_oth} are in the non-seeded target')
            selected_descr['predicted_other']['non_seeded_target'] = bias_indices_selected_oth

        ## Overall
        actual_tar = sum(train.y[selected_indices])
        print(f'All: Selected {len(selected_indices)} samples, with {actual_tar} target class')
        selected_descr['all'] = {}
        selected_descr['all']['selected'] = len(selected_indices)
        selected_descr['all']['target'] = actual_tar
        if biased:
            bias_indices_selected_all = len([i for i in selected_indices if i in bias_indices])
            print(f'of these {bias_indices_selected_all} are in the non-seeded target')
            selected_descr['all']['non_seeded_target'] = bias_indices_selected_all 

        return selected_indices


class NIHDAL_2(DiscriminativeActiveLearning_amended):

    """Similar to Discriminative Active Learning, but applied reweighting the pool.
    """

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):

        # Predict target or other for unlabelled data
        preds = active_learner.classifier.predict(train)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        # Describe predicted target
        target_indices_unlabeled_tar_count = sum(train.y[target_indices_unlabeled])
        print(f'There are {len(target_indices_unlabeled)} predicted target examples, of which {target_indices_unlabeled_tar_count} are actually target')
        if biased:
            bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in target_indices_unlabeled])
            print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')

        # Describe predicted other
        other_indices_unlabeled_tar_count = sum(train.y[other_indices_unlabeled])
        print(f'There are {len(other_indices_unlabeled)} predicted non-target examples, of which {other_indices_unlabeled_tar_count} are actually target')
        if biased:
            bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in other_indices_unlabeled])
            print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')

        # Create balanced pool
        half_pool_size = min(len(target_indices_unlabeled), len(other_indices_unlabeled))

        target_pool = np.random.choice(target_indices_unlabeled, half_pool_size, replace=False)
        other_pool = np.random.choice(other_indices_unlabeled, half_pool_size, replace=False)

        balanced_indices_unlabeled = np.concatenate((target_pool, other_pool)).astype(int)

        # Run DAL
        self._validate_query_input(balanced_indices_unlabeled, n)

        query_sizes = self._get_query_sizes(self.num_iterations, int(n))

        selected_indices = self.discriminative_active_learning(
            dataset,
            balanced_indices_unlabeled,
            indices_labeled,
            query_sizes
        )

        # Describe selected indices
        global selected_descr
        selected_descr = {}

        ## From pred target
        pred_tar_selected = [i for i in selected_indices if i in target_pool]
        pred_tar_actual_tar = sum(train.y[pred_tar_selected])
        print(f'Predicted target: Selected {len(pred_tar_selected)} samples, with {pred_tar_actual_tar} target class')
        selected_descr['predicted_target'] = {}
        selected_descr['predicted_target']['selected'] = len(pred_tar_selected)
        selected_descr['predicted_target']['target'] = pred_tar_actual_tar
        if biased:
            bias_indices_selected_tar = len([i for i in pred_tar_selected if i in bias_indices])
            print(f'of these {bias_indices_selected_tar} are in the non-seeded target')
            selected_descr['predicted_target']['non_seeded_target'] = bias_indices_selected_tar

        ## From pred other
        pred_oth_selected = [i for i in selected_indices if i in other_pool]
        pred_oth_actual_tar = sum(train.y[pred_oth_selected])
        print(f'Predicted non-target: Selected {len(pred_oth_selected)} samples, with {pred_oth_actual_tar} target class')
        selected_descr['predicted_other'] = {}
        selected_descr['predicted_other']['selected'] = len(pred_oth_selected)
        selected_descr['predicted_other']['target'] = pred_oth_actual_tar
        if biased:
            bias_indices_selected_oth = len([i for i in pred_oth_selected if i in bias_indices])
            print(f'of these {bias_indices_selected_oth} are in the non-seeded target')
            selected_descr['predicted_other']['non_seeded_target'] = bias_indices_selected_oth

        ## Overall
        actual_tar = sum(train.y[selected_indices])
        print(f'All: Selected {len(selected_indices)} samples, with {actual_tar} target class')
        selected_descr['all'] = {}
        selected_descr['all']['selected'] = len(selected_indices)
        selected_descr['all']['target'] = actual_tar
        if biased:
            bias_indices_selected_all = len([i for i in selected_indices if i in bias_indices])
            print(f'of these {bias_indices_selected_all} are in the non-seeded target')
            selected_descr['all']['non_seeded_target'] = bias_indices_selected_all 

        return selected_indices

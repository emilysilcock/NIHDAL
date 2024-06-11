import logging
import pickle

import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets
import numpy as np
import torch
from transformers import AutoTokenizer

import small_text
from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    DiscriminativeActiveLearning,
    random_initialization_balanced
)


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
        try:
            target_indices_unlabeled_tar_count = sum(train.y[target_indices_unlabeled])
            print(f'There are {len(target_indices_unlabeled)} predicted target examples, of which {target_indices_unlabeled_tar_count} are actually target')
            if biased:
                    bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in target_indices_unlabeled])
                    print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')
        except:
            pass

        # Describe predicted other
        try:
            other_indices_unlabeled_tar_count = sum(train.y[other_indices_unlabeled])
            print(f'There are {len(other_indices_unlabeled)} predicted non-target examples, of which {other_indices_unlabeled_tar_count} are actually target')
            if biased:
                    bias_indices_unlabeled_tar_count = len([i for i in bias_indices if i in other_indices_unlabeled])
                    print(f'of these {bias_indices_unlabeled_tar_count} are in the non-seeded target')
        except:
            pass

        # If there are not enough predicted targets
        if len(target_indices_unlabeled) <= n/2:
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
        elif len(other_indices_unlabeled) <= n/2:

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


def make_binary(dataset, target_labels):

    # Create mapping
    num_classes = dataset.features['label'].num_classes

    class_mapping = {lab: 0 for lab in range(num_classes)}
    
    for tl in target_labels:
        class_mapping[tl] = 1

    # Apply the mapping to change the labels
    binary_dataset = dataset.map(lambda example: {'label': class_mapping[example['label']]})

    # Update metadata
    new_features = datasets.Features({
        'text': binary_dataset.features['text'],
        'label': datasets.ClassLabel(names = ['merged', 'target'], num_classes=2)
        })
    binary_dataset = binary_dataset.cast(new_features)

    return binary_dataset


def make_imbalanced(dataset, indices_to_track=None):

    # Split dataset
    other_samples = dataset.filter(lambda example: example['label'] == 0)
    target_samples = dataset.filter(lambda example: example['label'] == 1)

    # Calculate the number of target samples to keep (1% of imbalanced dataset)
    other_samples_count = len(other_samples)
    imbalanced_total = other_samples_count/0.99
    target_count = int(imbalanced_total * 0.01)

    # Filter target samples to target number
    target_samples = target_samples.shuffle()
    target_samples_to_keep = target_samples.select(range(target_count))

    # Concat back together
    imbalanced_dataset = datasets.concatenate_datasets([target_samples_to_keep, other_samples])

    if indices_to_track:

        target_list = [i for i in target_samples_to_keep]

        tracked_indices = []
        for idx in indices_to_track:
            point = dataset[idx]
            if point in target_list:
                tracked_indices.append(target_list.index(dataset[idx]))

        return imbalanced_dataset, tracked_indices

    else:
        return imbalanced_dataset


def evaluate(active_learner, train, test):

    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    labelled_embeddings = active_learner.classifier.embed(train)
    test_embeddings = active_learner.classifier.embed(test)

    print(labelled_embeddings.shape)

    r = {
        'Train accuracy': accuracy_score(y_pred, train.y),
        'Test accuracy': accuracy_score(y_pred_test, test.y),
        'Train F1': f1_score(y_pred, train.y),
        'Test F1': f1_score(y_pred_test, test.y),
        'Train precision': precision_score(y_pred, train.y),
        'Test precision': precision_score(y_pred_test, test.y),
        'Train recall': recall_score(y_pred, train.y),
        'Test recall': recall_score(y_pred_test, test.y),
        'Test predictions': y_pred_test,
        'Test ground truth': test.y,
        'Test embeddings': test_embeddings,
        'Labelled data embeddings': labelled_embeddings,
        'Labelled data labels': train.y
    }

    return r


def random_initialization_biased(y, n_samples=10, non_sample=None):
    """Randomly draws half class 1, in a biased way, and half class 0. 

    Parameters
    ----------
    y : np.ndarray[int] or csr_matrix
        Labels to be used for stratification.
    n_samples :  int
        Number of samples to draw.

    Returns
    -------
    indices : np.ndarray[int]
        Indices relative to y.
    """

    expected_samples_per_class = np.floor(n_samples/2).astype(int)

    # Targets labels - don't sample from non_sample
    all_indices = [i for i, lab in enumerate(y) if lab == 1 and i not in non_sample]
    target_sample = random.sample(all_indices, expected_samples_per_class)

    # Non-target labels
    all_indices = [i for i, lab in enumerate(y) if lab == 0]
    other_sample = random.sample(all_indices, expected_samples_per_class)

    return np.array(target_sample + other_sample)


def initialize_active_learner_biased(active_learner, y_train, biased_indices):

    # simulates an initial labeling to warm-start the active learning process

    indices_initial = random_initialization_biased(y_train, n_samples=100, non_sample=biased_indices)
    active_learner.initialize(indices_initial, y_train[indices_initial])

    return indices_initial


def initialize_active_learner_balanced(active_learner, y_train):

    # simulates an initial labeling to warm-start the active learning process

    indices_initial = random_initialization_balanced(y_train, n_samples=100)
    active_learner.initialize(indices_initial, y_train[indices_initial])

    return indices_initial


def load_and_format_dataset(dataset_name, tokenization_model, target_labels=[0], biased=False):

    # Load data
    datasets_dict = {
        'isear': 'dalopeza98/isear-cleaned-dataset',
        'agnews': 'agnews'
    }

    raw_dataset = datasets.load_dataset(datasets_dict[dataset_name])

    if biased:
        unsampled_train_indices = [i for i, lab in enumerate(raw_dataset['train']['label']) if lab == target_labels[1]]

    # Reduce to two classes
    raw_dataset['train'] = make_binary(raw_dataset['train'], target_labels)
    raw_dataset['test'] = make_binary(raw_dataset['test'], target_labels)

    # Make target class 1% of the data
    if biased:
        raw_dataset['train'], bias_indices = make_imbalanced(raw_dataset['train'], indices_to_track=unsampled_train_indices)
    else:
        raw_dataset['train'] = make_imbalanced(raw_dataset['train'])

    raw_dataset['test'] = make_imbalanced(raw_dataset['test'])

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    num_classes = raw_dataset['train'].features['label'].num_classes
    lab_array = np.arange(num_classes)

    train_dat = TransformersDataset.from_arrays(raw_dataset['train']['text'],
                                            raw_dataset['train']['label'],
                                            tokenizer,
                                            max_length=100,
                                            target_labels=lab_array)
    test_dat = TransformersDataset.from_arrays(raw_dataset['test']['text'],
                                          raw_dataset['test']['label'],
                                          tokenizer,
                                          max_length=100,
                                          target_labels=lab_array)

    if biased:
        return train_dat, test_dat, bias_indices

    else:
        return train_dat, test_dat


def set_up_active_learner(transformer_model_name, active_learning_method):

    # Set up active learner
    num_classes = 2

    transformer_model = TransformerModelArguments(transformer_model_name)

    # hyperparameters = {
    #     'device': 'cuda',
    #     'mini_batch_size': 50,
    #     'model_selection': model_selection
    # }


    clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes,
                                                        kwargs=dict({'device': 'cuda',
                                                                    'mini_batch_size': 32,
                                                                    'num_epochs': 20,    ########
                                                                    'lr': 5e-5,    #######
                                                                    'class_weight': 'balanced'
                                                                    }))

    clf_factory_2 = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes,
                                                        kwargs=dict({'device': 'cuda',
                                                                    'mini_batch_size': 32,
                                                                    'num_epochs': 20,    ########
                                                                    'lr': 5e-5,    #######
                                                                    'class_weight': 'balanced'
                                                                    }))

    if active_learning_method == "DAL":
        query_strategy = DiscriminativeActiveLearning_amended(classifier_factory=clf_factory_2, num_iterations=10)
    elif active_learning_method == "NIHDAL":
        query_strategy = NIHDAL(classifier_factory=clf_factory_2, num_iterations=10)
    elif active_learning_method == "NIHDAL_simon":
        query_strategy = NIHDAL_2(classifier_factory=clf_factory_2, num_iterations=10)
    elif active_learning_method == "Random":
        query_strategy = small_text.query_strategies.strategies.RandomSampling()
    elif active_learning_method == "Least Confidence":
        query_strategy = small_text.LeastConfidence()
    elif active_learning_method == "Prediction Entropy":
        query_strategy = small_text.PredictionEntropy()
    elif active_learning_method == "BALD":
       query_strategy = small_text.query_strategies.bayesian.BALD()
    elif active_learning_method == "EGL":
        query_strategy = small_text.ExpectedGradientLength(num_classes=2)
    elif active_learning_method == "BADGE":
        query_strategy = small_text.integrations.pytorch.query_strategies.strategies.BADGE(num_classes=2)
    elif active_learning_method == "Core Set":
        query_strategy = small_text.query_strategies.coresets.GreedyCoreset()
    elif active_learning_method == "Contrastive":
        query_strategy = small_text.query_strategies.strategies.ContrastiveActiveLearning()
    else:
        raise ValueError(f"Active Learning method {active_learning_method} is unknown")

    # if "DAL" in active_learning_method:

    #     ## DAL paper they find that early stopping is important and they use 0.98 on accuracy
    #     early_stopping = small_text.training.early_stopping.EarlyStopping(small_text.training.metrics.Metric('train_acc', lower_is_better=False), threshold=0.90)

    #     a_learner = PoolBasedActiveLearner(
    #         clf_factory,
    #         query_strategy,
    #         train,
    #         reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining
    #         fit_kwargs={'early_stopping': early_stopping}
    #     )

    # else:
    a_learner = PoolBasedActiveLearner(
        clf_factory,
        query_strategy,
        train,
        reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining
    )

    return a_learner


def active_learning_loop(active_learner, train, test, num_queries, bias, selected_descr):

    # Initialise with first sample
    if bias:
        indices_labeled = initialize_active_learner_biased(active_learner, train.y, bias)
    else:
        indices_labeled = initialize_active_learner_balanced(active_learner, train.y)

    print(f'Initial sample contains {sum(train.y[indices_labeled])} target class')
    if bias:
        in_bias = [i for i in indices_labeled if i in bias]
        print(f'Initial sample contains {len(in_bias)} from non-seeded target class')


    results = []
    results.append(evaluate(active_learner, train[indices_labeled], test))
        
    for i in range(num_queries):

        # Query samples to label
        indices_queried = active_learner.query(num_samples=100)

        # Simulate labelling
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('---------------')
        print(f'Iteration #{i} ({len(indices_labeled)} samples)')
        res = evaluate(active_learner, train[indices_labeled], test)

        if als not in ['NIHDAL', 'NIHDAL_simon']:

            selected_descr = {
                'all': {
                    'selected': len(indices_queried),
                    'target': int(sum(y))
                }
            }

            if biased:
                selected_descr['all']['non_seeded_target'] = len([i for i in indices_queried if i in bias])

        res['counts'] = selected_descr

        results.append(res)

    return results


if __name__ == '__main__':

    datasets.logging.set_verbosity_error()
    datasets.logging.get_verbosity = lambda: logging.NOTSET

    transformer_model_name = 'distilroberta-base'

    for ds in ['isear']:
        for biased in [False, True]:
            for als in ["Random", "Least Confidence", "BALD", "BADGE", "DAL", "Core Set", 'NIHDAL', 'NIHDAL_simon']: #"Contrastive",

                print(f'****************{als}**********************')

                # Set seed
                for seed in [12731]:  # 42, 12731, 65372, 97, 163

                    print(f'#################{seed}##################')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    selected_descr = None

                    # Load data
                    if biased:
                        train, test, bias_indices = load_and_format_dataset(
                            dataset_name=ds,
                            tokenization_model=transformer_model_name,
                            target_labels=[0, 1],
                            biased=True
                        )

                    else:
                        train, test = load_and_format_dataset(
                            dataset_name=ds,
                            tokenization_model=transformer_model_name,
                            target_labels=[0]
                        )
                        bias_indices = None

                    active_learner = set_up_active_learner(transformer_model_name, active_learning_method=als)

                    results = active_learning_loop(active_learner, train, test, num_queries=10, bias=bias_indices, selected_descr=selected_descr)

                    if biased:
                        with open(f'results/{ds}_{als}_results_{seed}_biased_new.pkl', 'wb') as f:
                            pickle.dump(results, f)

                    else:
                        with open(f'results/{ds}_{als}_results_{seed}_unbiased.pkl', 'wb') as f:
                            pickle.dump(results, f)

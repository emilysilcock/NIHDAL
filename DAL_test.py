import logging
import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets
import numpy as np
import torch
from transformers import AutoTokenizer

import small_text
from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    PredictionEntropy,
    DiscriminativeActiveLearning,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
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

    """Similar to Discriminative Active Learning, but applied on the predicted positives and 
     negatives separately. 
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

        print(len(target_indices_unlabeled), len(other_indices_unlabeled), len(target_indices_labeled), len(other_indices_labeled))

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

        all_indices = np.concatenate((target_indices, other_indices)).astype(int)
        return all_indices


class NIHDAL_2(DiscriminativeActiveLearning_amended):

    """Similar to Discriminative Active Learning, but applied on the predicted positives and 
     negatives separately. 
    """

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):

        # Predict target or other for unlabelled data
        preds = active_learner.classifier.predict(train)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        # Describe predicted target
        target_indices_unlabeled_pos_count = sum([train.y[i] for i in target_indices_unlabeled])
        print(f'There are {len(target_indices_unlabeled)} predicted target examples, of which {target_indices_unlabeled_pos_count} are actually target')

        # Describe predicted other
        other_indices_unlabeled_pos_count = sum([train.y[i] for i in other_indices_unlabeled])
        print(f'There are {len(other_indices_unlabeled)} predicted non-target examples, of which {other_indices_unlabeled_pos_count} are actually target')

        # Create balanced pool
        half_pool_size = min(len(target_indices_unlabeled), len(other_indices_unlabeled))

        target_pool = np.random.choice(target_indices_unlabeled, half_pool_size, replace=False)
        other_pool = np.random.choice(other_indices_unlabeled, half_pool_size, replace=False)

        balanced_indices_unlabeled = np.concatenate((target_pool, other_pool)).astype(int).shuffle()

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
        ## From pred pos
        pred_pos_selected = [i for i in selected_indices if i in target_pool]
        pred_pos_actual_pos = sum(train.y[pred_pos_selected])
        print(f'Predicted positives: Selected {len(pred_pos_selected)} samples, with {pred_pos_actual_pos} target class')

        ## From pred neg
        pred_neg_selected = [i for i in selected_indices if i in other_pool]
        pred_neg_actual_pos = sum(train.y[pred_neg_selected])
        print(f'Predicted negatives: Selected {len(pred_neg_selected)} samples, with {pred_neg_actual_pos} target class')

        ## Overall 
        actual_pos = sum(train.y[selected_indices])
        print(f'All: Selected {len(selected_indices)} samples, with {actual_pos} target class')

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

    r = {
        'Train accuracy': accuracy_score(y_pred, train.y),
        'Test accuracy': accuracy_score(y_pred_test, test.y),
        'Train F1': f1_score(y_pred, train.y),
        'Test F1': f1_score(y_pred_test, test.y),
        'Train precision': precision_score(y_pred, train.y),
        'Test precision': precision_score(y_pred_test, test.y),
        'Train recall': recall_score(y_pred, train.y),
        'Test recall': recall_score(y_pred_test, test.y)
    }

    print(json.dumps(r, indent=4))

    return r


def initialize_active_learner(active_learner, y_train):

    # simulates an initial labeling to warm-start the active learning process

    indices_initial = random_initialization_balanced(y_train, n_samples=100)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


def load_and_format_dataset(dataset_name, tokenization_model, target_labels=[0]):

    # Load data
    raw_dataset = datasets.load_dataset(dataset_name)

    # Reduce to two classes
    raw_dataset['train'] = make_binary(raw_dataset['train'], target_labels)
    raw_dataset['test'] = make_binary(raw_dataset['test'], target_labels)

    # Make target class 1% of the daya
    raw_dataset['train'] = make_imbalanced(raw_dataset['train'])
    raw_dataset['test'] = make_imbalanced(raw_dataset['test'])

    # Tokenize data 
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    num_classes = raw_dataset['train'].features['label'].num_classes
    lab_array = np.arange(num_classes)

    train = TransformersDataset.from_arrays(raw_dataset['train']['text'],
                                            raw_dataset['train']['label'],
                                            tokenizer,
                                            max_length=100,
                                            target_labels=lab_array)
    test = TransformersDataset.from_arrays(raw_dataset['test']['text'],
                                          raw_dataset['test']['label'],
                                          tokenizer,
                                          max_length=100,
                                          target_labels=lab_array)
    
    return train, test


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
        # query_strategy = NIHDAL(classifier_factory=clf_factory_2, num_iterations=10)
        query_strategy = NIHDAL_2(classifier_factory=clf_factory_2, num_iterations=10)
    elif active_learning_method == "Random":
        query_strategy = small_text.query_strategies.strategies.RandomSampling()
    elif active_learning_method == "Least Confidence":
        query_strategy = small_text.LeastConfidence()
    elif active_learning_method == "Prediction Entropy":
        query_strategy = small_text.PredictionEntropy()
    elif active_learning_method == "BALD":
       query_strategy = small_text.query_strategies.bayesian.BALD()
    elif active_learning_method == "Expected Gradient Length":
        query_strategy = small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength(num_classes=2, device='cuda')
    elif active_learning_method == "BADGE":
        query_strategy = small_text.integrations.pytorch.query_strategies.strategies.BADGE(num_classes=2)
    elif active_learning_method == "Core Set":
        query_strategy = small_text.query_strategies.coresets.GreedyCoreset()
    elif active_learning_method == "Contrastive":
        query_strategy = small_text.query_strategies.strategies.ContrastiveActiveLearning()
    else:
        raise ValueError(f"Active Learning method {active_learning_method} is unknown")

    a_learner = PoolBasedActiveLearner(
        clf_factory,
        query_strategy,
        train,
        reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining
    )

    return a_learner


def active_learning_loop(active_learner, train, test, num_queries):

    # Initialise with first sample 
    indices_labeled = initialize_active_learner(active_learner, train.y)

    print(f'Initial sample contains {sum(train.y[indices_labeled])} target class')

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
        results.append(evaluate(active_learner, train[indices_labeled], test))

    return results


if __name__ == '__main__':

    datasets.logging.set_verbosity_error()

    # disables the progress bar for notebooks: https://github.com/huggingface/datasets/issues/2651
    datasets.logging.get_verbosity = lambda: logging.NOTSET

    # Set seed
    for seed in [42, 12731, 65372]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # transformer_model_name = 'bert-base-uncased'
        transformer_model_name = 'distilroberta-base'

        # Load data
        test, train = load_and_format_dataset(
            dataset_name='ag_news',
            tokenization_model=transformer_model_name,
            target_labels=[0]
        )

        active_learner = set_up_active_learner(transformer_model_name, active_learning_method="NIHDAL")

        results = active_learning_loop(active_learner, train, test, num_queries=10)

    # Todo:
    # - Minibatch size
    # - biased initial seed
    # - Unlabelled factor
    # - Early stopping
    # - Add counts of target and non-target to results

import logging

from sklearn.metrics import accuracy_score, f1_score
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
    
    test_acc = accuracy_score(y_pred_test, test.y)

    print('Train accuracy: {:.2f}'.format(accuracy_score(y_pred, train.y)))
    print('Test accuracy: {:.2f}'.format(test_acc))
    print('Train F1: {:.2f}'.format(f1_score(y_pred, train.y, average="weighted")))      ############################
    print('Test F1: {:.2f}'.format(f1_score(y_pred_test, test.y, average="weighted")))   ############################
    
    return test_acc


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
                                            max_length=60,
                                            target_labels=lab_array)
    test = TransformersDataset.from_arrays(raw_dataset['test']['text'], 
                                          raw_dataset['test']['label'],
                                          tokenizer,
                                          max_length=60,
                                          target_labels=lab_array)
    
    return train, test


def set_up_active_learner(transformer_model_name, active_learning_method):

    # Set up active learner 
    num_classes = 2

    transformer_model = TransformerModelArguments(transformer_model_name)

    clf_factory = TransformerBasedClassificationFactory(transformer_model, 
                                                        num_classes, 
                                                        kwargs=dict({'device': 'cuda', 
                                                                    'mini_batch_size': 32,
                                                                    'class_weight': 'balanced'
                                                                    }))

    clf_factory_2 = TransformerBasedClassificationFactory(transformer_model, 
                                                        num_classes, 
                                                        kwargs=dict({'device': 'cuda', 
                                                                    'mini_batch_size': 32,
                                                                    'class_weight': 'balanced'
                                                                    }))

    if active_learning_method == "DAL":
        query_strategy = DiscriminativeActiveLearning(classifier_factory=clf_factory_2, num_iterations=10)
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

    a_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)

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
        lt = len([i for i in y if i == 1])
        lo = len([i for i in y if i == 0])
        print(f'Selected {lt} samples of target class and {lo} of non-target class for labelling')

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
    seed = 2022
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

    active_learner = set_up_active_learner(transformer_model_name, active_learning_method="DAL")

    results = active_learning_loop(active_learner, train, test, num_queries=10)

    # Todo:
    # - DAL using classification model
    # - NIHDAL
    # - biased initial seed
    # - max token length during tokenisation
    # - Save all scores, not just accuracy
    # - Minibatch size
    # - Learning rate
    # - Number of epochs
    # - Unlabelled factor
    # - Early stopping
    # - Reuse model = False

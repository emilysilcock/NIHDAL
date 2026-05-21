import logging
import pickle

import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets
import numpy as np
import torch

import tensorflow_datasets as tfds
from transformers import AutoTokenizer

import small_text
from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    DiscriminativeActiveLearning,
    DiscriminativeRepresentationLearning,
    random_initialization_balanced
)

# Own query method classes ------------------------------------------------------------------------


# Functions for creating the data ----------------------------------------------------------------

def load_and_format_dataset(dataset_name, tokenization_model, target_labels=[0], biased_labels=[]):

    # Load data
    datasets_dict = {
        'ag_news':
            {
                'name': 'ag_news',
                'text_name': 'text',
                'label_name': 'label'
            },
        'trec-10':
            {
                'name': 'trec',
                'text_name': 'text',
                'label_name': 'label-coarse'
            }
    }

    if dataset_name == "trec-10":
        tf_dataset = tfds.load('trec')

        # Convert to hf dataset
        df_train = tfds.as_dataframe(tf_dataset['train'])
        df_test = tfds.as_dataframe(tf_dataset['test'])

        for k in df_train:
            if k not in [datasets_dict[dataset_name]['label_name'], datasets_dict[dataset_name]['text_name']]:
                del df_test[k]
                del df_train[k]

        num_classes = len(set(df_test[datasets_dict[dataset_name]['label_name']]))
        print(num_classes)

        names = [f"class_{i}" for i in range(num_classes)]
        print(names)

        class_label = datasets.ClassLabel(num_classes=num_classes, names=names)

        features = datasets.Features({
            datasets_dict[dataset_name]['text_name']: datasets.Value("string"),
            datasets_dict[dataset_name]['label_name']: class_label
            })

        raw_dataset = datasets.DatasetDict({
            'train': datasets.Dataset.from_pandas(df_train, features=features),
            'test': datasets.Dataset.from_pandas(df_test, features=features)
        })

    elif dataset_name == 'ag_news':
        raw_dataset = datasets.load_dataset(datasets_dict[dataset_name]['name'])

    # Rename text column if necessary
    if datasets_dict[dataset_name]['text_name'] != 'text':
        raw_dataset = raw_dataset.rename_column(datasets_dict[dataset_name]['text_name'], 'text')

    # Rename label column if necessary
    if datasets_dict[dataset_name]['label_name'] != 'label':
        raw_dataset = raw_dataset.rename_column(datasets_dict[dataset_name]['label_name'], 'label')

    # Keep track of unlabelled class
    if biased_labels:
        unsampled_train_indices = [i for i, lab in enumerate(raw_dataset['train']['label']) if lab in biased_labels]

    # Reduce to two classes
    raw_dataset['train'] = make_binary(raw_dataset['train'], target_labels)
    raw_dataset['test'] = make_binary(raw_dataset['test'], target_labels)

    # Make target class 1% of the data
    if biased_labels:
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

    if biased_labels:
        return train_dat, test_dat, bias_indices

    else:
        return train_dat, test_dat

def make_binary(dataset, target_labels):

    # target_labels contains the original label values that are to become target labels,
    # all the others are then 0

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
    print(f'There are {target_count} target examples left in the dataset')

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

# Functions for active learning ---------------------------------------------------------------

def set_up_active_learner(transformer_model_name, active_learning_method,
                          train_dataset,
                          TransformerBasedClassificationFactory_kwargs = dict({'device': 'cuda',
                                                                    'mini_batch_size': 32,
                                                                    'num_epochs': 20,    ########
                                                                    'lr': 5e-5,    #######
                                                                    'class_weight': 'balanced'
                                                                    })):

    # Set up active learner
    num_classes = 2

    transformer_model = TransformerModelArguments(transformer_model_name)


    clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes,
                                                        kwargs=TransformerBasedClassificationFactory_kwargs)

    clf_factory_2 = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes,
                                                        kwargs=TransformerBasedClassificationFactory_kwargs)


    # Setting the query method
    if active_learning_method == "DAL":
        # query_strategy = DiscriminativeActiveLearning_amended(classifier_factory=clf_factory_2, num_iterations=10)
        query_strategy = DiscriminativeRepresentationLearning(num_iterations=10, selection='greedy')
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

    # Initialize the active learner
    a_learner = PoolBasedActiveLearner(
        clf_factory,
        query_strategy,
        train_dataset,
        reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining
    )

    return a_learner

def random_initialization_biased(y, n_samples=10, non_sample=None):
    """Randomly draws half class 1, in a biased way, and half class 0.

    Parameters
    ----------
    y : np.ndarray[int] or csr_matrix
        Labels to be used for stratification.
    n_samples :  int
        Number of samples to draw.
    non_sample :
        target indices from which not to sample for initialization

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

def initialize_active_learner(active_learner, y_train, biased_indices = []):

    # simulates an initial labeling to warm-start the active learning process
    if biased_indices:
        indices_initial = random_initialization_biased(y_train, n_samples=100, non_sample=biased_indices)
    else:
        indices_initial = random_initialization_balanced(y_train, n_samples=100)

    # active_learner.initialize_data(indices_initial, y_train[indices_initial])
    active_learner.initialize(indices_initial, y_train[indices_initial])

    return indices_initial

def evaluate(active_learner, train, test):

    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    labelled_embeddings = active_learner.classifier.embed(train)
    test_embeddings = active_learner.classifier.embed(test)

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

    print('Test accuracy:', r['Test accuracy'], 'Test F1:', r['Test F1'])

    return r

def active_learning_loop(active_learner, train, test, num_queries, bias, selected_descr, active_learning_method):

    # Initialise with first sample
    if bias:
        indices_labeled = initialize_active_learner(active_learner, train.y, bias)
    else:
        indices_labeled = initialize_active_learner(active_learner, train.y)

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

        if active_learning_method not in ['NIHDAL', 'NIHDAL_simon']:

            selected_descr = {
                'all': {
                    'selected': len(indices_queried),
                    'target': int(sum(y))
                }
            }

            if bias:
                selected_descr['all']['non_seeded_target'] = len([i for i in indices_queried if i in bias])

        res['counts'] = selected_descr

        print(selected_descr)

        results.append(res)

    return results

# Main body -----------------------------------------------------------

if __name__ == '__main__':

    datasets.logging.set_verbosity_error()
    datasets.logging.get_verbosity = lambda: logging.NOTSET

    transformer_model_name = 'distilroberta-base'

    for ds in ['ag_news']:
        for biased in [True]:
            # for als in ["Random", "Least Confidence", "BALD", "BADGE", "DAL", "Core Set", 'NIHDAL', 'NIHDAL_simon']: #"Contrastive",
            for als in ['Least Confidence']:

                print(f'****************{als}**********************')

                # Set seed
                for seed in [42]:  # 42, 12731, 65372, 97, 163

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
                            biased_labels=[1]
                        )

                    else:
                        train, test = load_and_format_dataset(
                            dataset_name=ds,
                            tokenization_model=transformer_model_name,
                            target_labels=[0]
                        )
                        bias_indices = None

                    active_learner = set_up_active_learner(transformer_model_name, active_learning_method=als, train_dataset = train)

                    results = active_learning_loop(active_learner, train, test, num_queries=10, bias=bias_indices, selected_descr=selected_descr,
                                                   active_learning_method=als)

                    if biased:
                        with open(f'/n/holyscratch01/economics/esilcock/NIHDAL_results/{ds}_{als}_results_{seed}_biased_new.pkl', 'wb') as f:
                            pickle.dump(results, f)

                    else:
                        with open(f'/n/holyscratch01/economics/esilcock/NIHDAL_results/{ds}_{als}_results_{seed}_unbiased.pkl', 'wb') as f:
                            pickle.dump(results, f)

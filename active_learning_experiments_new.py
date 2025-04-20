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
    DiscriminativeRepresentationLearning,
    random_initialization_balanced
)

# Imports for amended classes -----------------------------------------

# import numpy.typing as npt

# from typing import Union

# from scipy.sparse import csr_matrix
# from scipy.special import softmax

# from small_text.classifiers import Classifier
# from small_text.data import Dataset
from small_text.query_strategies.strategies import DiscriminativeActiveLearning

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.query_strategies import (
    constraints,
    QueryStrategy,
    EmbeddingBasedQueryStrategy)
from small_text.utils.clustering import init_kmeans_plusplus_safe
from small_text.utils.context import build_pbar_context
from small_text.utils.data import list_length

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.amp import GradScaler  # pyright: ignore
    from torch.nn import BCEWithLogitsLoss
    from torch.nn.utils import clip_grad_norm_  # pyright: ignore

    from torch.optim import Adam

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.models.mlp import MLP

    from small_text.integrations.pytorch.utils.misc import _assert_layer_exists
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


# Own query method classes ------------------------------------------------------------------------

class PretrainedDiscriminativeActiveLearning(QueryStrategy):
    """Discriminative Active Learning that initializes the discriminative transformer classifier
    with weights from the main transformer model.
    """

    LABEL_LABELED_POOL = 0
    LABEL_UNLABELED_POOL = 1

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10, pbar='tqdm'):
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
        indices = self._discriminative_active_learning(clf, dataset, indices_unlabeled, indices_labeled,
                                                      query_sizes)
        return indices

    def _discriminative_active_learning(self, clf, dataset, indices_unlabeled, indices_labeled, query_sizes):
        indices = np.array([], dtype=indices_labeled.dtype)
        indices_unlabeled_copy = np.copy(indices_unlabeled)
        indices_labeled_copy = np.copy(indices_labeled)

        with build_pbar_context(len(query_sizes)) as pbar:
            for q in query_sizes:
                indices_most_confident = self._train_and_get_most_confident(clf, dataset,
                                                                         indices_unlabeled_copy,
                                                                         indices_labeled_copy, q)

                indices = np.append(indices, indices_unlabeled_copy[indices_most_confident])
                indices_labeled_copy = np.append(indices_labeled_copy,
                                              indices_unlabeled_copy[indices_most_confident])
                indices_unlabeled_copy = np.delete(indices_unlabeled_copy, indices_most_confident)
                pbar.update(1)

        return indices

    def _train_and_get_most_confident(self, clf, ds, indices_unlabeled, indices_labeled, q):
        if self.clf_ is not None:
            del self.clf_
                
        # Create a new binary classifier from the factory
        original_num_classes = self.classifier_factory.num_classes
        self.classifier_factory.num_classes = 2
        discr_clf = self.classifier_factory.new()
        self.classifier_factory.num_classes = original_num_classes
        
        # Create the discriminative dataset first
        num_unlabeled = min(indices_labeled.shape[0] * self.unlabeled_factor,
                        indices_unlabeled.shape[0])

        indices_unlabeled_sub = np.random.choice(indices_unlabeled,
                                            num_unlabeled,
                                            replace=False)

        ds_discr = DiscriminativeActiveLearning.get_relabeled_copy(ds,
                                                                indices_unlabeled_sub,
                                                                indices_labeled)
        
        # Try to initialize the model if needed
        # Many transformer models initialize on first use
        if hasattr(discr_clf, 'initialize'):
            discr_clf.initialize()
        
        # # First make sure the discriminative classifier works on its own
        # # This will ensure the model is properly initialized
        # discr_clf.fit(ds_discr)
        
        # Now try copying weights from the main model if possible
        if hasattr(clf, 'model') and hasattr(discr_clf, 'model') and \
        clf.model is not None and discr_clf.model is not None:
            try:
                # Get state dictionaries
                main_state_dict = clf.model.state_dict()
                discr_state_dict = discr_clf.model.state_dict()
                
                # Copy all matching parameters except the classification head
                for name, param in main_state_dict.items():
                    # Skip classification head parameters and mismatched shapes
                    if 'classifier' not in name and 'head' not in name and \
                    name in discr_state_dict and \
                    discr_state_dict[name].shape == param.shape:
                        discr_state_dict[name].copy_(param)
                
                # Load the updated state dict
                discr_clf.model.load_state_dict(discr_state_dict)
                
                # Retrain with the copied weights
                self.clf_ = discr_clf.fit(ds_discr)
            except Exception as e:
                # Fall back to the already trained classifier if weight transfer fails
                self.clf_ = discr_clf
                print(f"Weight transfer failed, using standard initialization. Error: {e}")
        else:
            self.clf_ = discr_clf

        # Get predictions
        proba = self.clf_.predict_proba(ds[indices_unlabeled])
        proba = proba[:, self.LABEL_UNLABELED_POOL]

        # Return instances which most likely belong to the "unlabeled" class
        return np.argpartition(-proba, q)[:q]

    def __str__(self):
        return f'PretrainedDiscriminativeActiveLearning(classifier_factory={str(self.classifier_factory)}, ' \
               f'num_iterations={self.num_iterations}, unlabeled_factor={self.unlabeled_factor})'


class NIHDAL(QueryStrategy):
    """Discriminative Active Learning that runs twice - once for predicted positives and once for predicted negatives.
    Each run initializes the discriminative transformer classifier with weights from the main transformer model.
    """

    LABEL_LABELED_POOL = 0
    LABEL_UNLABELED_POOL = 1

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10, pbar='tqdm', split_ratio=0.5):
        self.classifier_factory = classifier_factory
        self.num_iterations = num_iterations
        self.unlabeled_factor = unlabeled_factor
        self.pbar = pbar
        self.clf_positive_ = None
        self.clf_negative_ = None
        self.split_ratio = split_ratio  # Proportion of samples to select from positive predictions

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        # Get predictions from the original classifier for the unlabeled data
        proba = clf.predict_proba(dataset[indices_unlabeled])
        
        # Split the unlabeled data into predicted positives and negatives
        # Class 1 is typically the positive class
        predicted_positive_mask = np.argmax(proba, axis=1) == 1
        
        # Get the indices for predicted positives and negatives
        indices_predicted_positive = indices_unlabeled[predicted_positive_mask]
        indices_predicted_negative = indices_unlabeled[~predicted_positive_mask]
        
        print(f"Unlabeled split: {len(indices_predicted_positive)} predicted positives, {len(indices_predicted_negative)} predicted negatives")
        
        # Calculate how many samples to select from each group
        n_positive = max(1, int(n * self.split_ratio))
        n_negative = n - n_positive
        
        # Adjust if one of the groups doesn't have enough samples
        if len(indices_predicted_positive) < n_positive:
            n_positive = len(indices_predicted_positive)
            n_negative = n - n_positive
        elif len(indices_predicted_negative) < n_negative:
            n_negative = len(indices_predicted_negative)
            n_positive = n - n_negative
            
        print(f"Selecting {n_positive} from positives, {n_negative} from negatives")
        
        # Run discriminative active learning on each group
        query_sizes_positive = DiscriminativeActiveLearning._get_query_sizes(self.num_iterations, n_positive)
        query_sizes_negative = DiscriminativeActiveLearning._get_query_sizes(self.num_iterations, n_negative)
        
        indices_positive = self._discriminative_active_learning(
            clf, dataset, indices_predicted_positive, indices_labeled, query_sizes_positive, is_positive=True
        )
        
        indices_negative = self._discriminative_active_learning(
            clf, dataset, indices_predicted_negative, indices_labeled, query_sizes_negative, is_positive=False
        )
        
        # Combine the results
        indices = np.concatenate([indices_positive, indices_negative])
        
        return indices

    def _discriminative_active_learning(self, clf, dataset, indices_unlabeled, indices_labeled, query_sizes, is_positive=True):
        indices = np.array([], dtype=indices_labeled.dtype)
        indices_unlabeled_copy = np.copy(indices_unlabeled)
        indices_labeled_copy = np.copy(indices_labeled)

        with build_pbar_context(len(query_sizes)) as pbar:
            for q in query_sizes:
                indices_most_confident = self._train_and_get_most_confident(
                    clf, dataset, indices_unlabeled_copy, indices_labeled_copy, q, is_positive
                )

                indices = np.append(indices, indices_unlabeled_copy[indices_most_confident])
                indices_labeled_copy = np.append(indices_labeled_copy,
                                              indices_unlabeled_copy[indices_most_confident])
                indices_unlabeled_copy = np.delete(indices_unlabeled_copy, indices_most_confident)
                pbar.update(1)

        return indices

    def _train_and_get_most_confident(self, clf, ds, indices_unlabeled, indices_labeled, q, is_positive=True):
        # Reset the appropriate classifier based on which pool we're working with
        if is_positive and self.clf_positive_ is not None:
            del self.clf_positive_
        elif not is_positive and self.clf_negative_ is not None:
            del self.clf_negative_
                
        # Create a new binary classifier from the factory
        original_num_classes = self.classifier_factory.num_classes
        self.classifier_factory.num_classes = 2
        discr_clf = self.classifier_factory.new()
        self.classifier_factory.num_classes = original_num_classes
        
        # Create the discriminative dataset first
        num_unlabeled = min(indices_labeled.shape[0] * self.unlabeled_factor,
                        indices_unlabeled.shape[0])

        indices_unlabeled_sub = np.random.choice(indices_unlabeled,
                                            num_unlabeled,
                                            replace=False)

        ds_discr = DiscriminativeActiveLearning.get_relabeled_copy(ds,
                                                                indices_unlabeled_sub,
                                                                indices_labeled)
        
        # Try to initialize the model if needed
        if hasattr(discr_clf, 'initialize'):
            discr_clf.initialize()
        
        # Try copying weights from the main model if possible
        if hasattr(clf, 'model') and hasattr(discr_clf, 'model') and \
        clf.model is not None and discr_clf.model is not None:
            try:
                # Get state dictionaries
                main_state_dict = clf.model.state_dict()
                discr_state_dict = discr_clf.model.state_dict()
                
                # Copy all matching parameters except the classification head
                for name, param in main_state_dict.items():
                    # Skip classification head parameters and mismatched shapes
                    if 'classifier' not in name and 'head' not in name and \
                    name in discr_state_dict and \
                    discr_state_dict[name].shape == param.shape:
                        discr_state_dict[name].copy_(param)
                
                # Load the updated state dict
                discr_clf.model.load_state_dict(discr_state_dict)
                
                # Retrain with the copied weights
                trained_clf = discr_clf.fit(ds_discr)
                
                # Store the classifier in the appropriate attribute
                if is_positive:
                    self.clf_positive_ = trained_clf
                else:
                    self.clf_negative_ = trained_clf
                    
            except Exception as e:
                # Fall back to the already trained classifier if weight transfer fails
                if is_positive:
                    self.clf_positive_ = discr_clf
                else:
                    self.clf_negative_ = discr_clf
                print(f"Weight transfer failed, using standard initialization. Error: {e}")
        else:
            if is_positive:
                self.clf_positive_ = discr_clf
            else:
                self.clf_negative_ = discr_clf

        # Use the appropriate classifier for predictions
        current_clf = self.clf_positive_ if is_positive else self.clf_negative_
        
        # Get predictions
        proba = current_clf.predict_proba(ds[indices_unlabeled])
        proba = proba[:, self.LABEL_UNLABELED_POOL]

        # Return instances which most likely belong to the "unlabeled" class
        return np.argpartition(-proba, q)[:q]

    def __str__(self):
        return f'NIHDAL(classifier_factory={str(self.classifier_factory)}, ' \
               f'num_iterations={self.num_iterations}, unlabeled_factor={self.unlabeled_factor}, ' \
               f'split_ratio={self.split_ratio})'

# Functions for creating the data ----------------------------------------------------------------

def load_and_format_dataset(dataset_name, tokenization_model, target_labels=[0], biased_labels=[]):

    # Load data
    datasets_dict = {
        'ag_news':
            {
                'name': 'ag_news',
                'text_name': 'text',
                'label_name': 'label'
            }
    }

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
                                                                    'class_weight': 'balanced',
                                                                    'cache_dir': '/n/netscratch/economics/Lab/esilcock/nihdal_results/cache'
                                                                    })):

    # Set up active learner
    num_classes = 2

    transformer_model = TransformerModelArguments(transformer_model_name)


    clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes,
                                                        classification_kwargs=TransformerBasedClassificationFactory_kwargs)


    # Setting the query method
    if active_learning_method == "DAL1":
        query_strategy = DiscriminativeActiveLearning(clf_factory, num_iterations=10)
    elif active_learning_method == "DAL2":
        query_strategy = PretrainedDiscriminativeActiveLearning(clf_factory, num_iterations=10)
    elif active_learning_method == "DAL3":
        query_strategy = DiscriminativeRepresentationLearning(num_iterations=10, selection='greedy')
    elif active_learning_method == "NIHDAL":
        query_strategy = NIHDAL(clf_factory, num_iterations=10)
    elif active_learning_method == "NIHDAL":
        query_strategy = NIHDAL(classifier_factory=clf_factory, num_iterations=10)
    # elif active_learning_method == "NIHDAL_simon":
    #     query_strategy = NIHDAL_2(classifier_factory=clf_factory_2, num_iterations=10)
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
            for als in ['DAL2', 'NIHDAL', 'Random']:

                print(f'****************{als}**********************')

                # Set seed
                for seed in [42]:  # 42, 12731, 65372, 97, 163
                # for seed in [42, 12731]:  # 42, 12731, 65372, 97, 163

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

                    results = active_learning_loop(active_learner, train, test, num_queries=3, bias=bias_indices, selected_descr=selected_descr,
                                                   active_learning_method=als)

                    if biased:
                        with open(f'/n/netscratch/economics/Lab/esilcock/nihdal_results/{ds}_{als}_results_{seed}_biased_new.pkl', 'wb') as f:
                            pickle.dump(results, f)

                    else:
                        with open(f'/n/netscratch/economics/Lab/esilcock/nihdal_results/{ds}_{als}_results_{seed}_unbiased.pkl', 'wb') as f:
                            pickle.dump(results, f)

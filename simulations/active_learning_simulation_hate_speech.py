import logging
import pickle
import os

import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
)

# Imports for amended classes -----------------------------------------

from small_text.query_strategies.strategies import DiscriminativeActiveLearning

from small_text.query_strategies import QueryStrategy
from small_text.utils.context import build_pbar_context


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
                # Fit the model from scratch if weight transfer fails
                discr_clf.fit(ds_discr)
                self.clf_ = discr_clf
                print(f"Weight transfer failed, training from scratch. Error: {e}")
        else:
            print("No weights to transfer, training from scratch")
            discr_clf.fit(ds_discr)
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
    Each run uses PretrainedDiscriminativeActiveLearning on the separate splits.
    """

    LABEL_LABELED_POOL = 0
    LABEL_UNLABELED_POOL = 1

    def __init__(self, classifier_factory, num_iterations=10, unlabeled_factor=10, pbar='tqdm', 
                 split_ratio=0.5):
        self.classifier_factory = classifier_factory
        self.num_iterations = num_iterations
        self.unlabeled_factor = unlabeled_factor
        self.pbar = pbar
        self.split_ratio = split_ratio
        # Create the base strategy that we'll reuse
        self.dal_strategy = PretrainedDiscriminativeActiveLearning(
            classifier_factory,
            num_iterations=num_iterations,
            unlabeled_factor=unlabeled_factor,
            pbar=pbar
        )

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
        
        # Always determine the split (how many samples to select from each group)
        # Handle special cases where one class has zero samples
        if len(indices_predicted_positive) == 0:
            print("No predicted positive samples found. Selecting all samples from negative class.")
            n_positive = 0
            n_negative = min(n, len(indices_predicted_negative))
            
        elif len(indices_predicted_negative) == 0:
            print("No predicted negative samples found. Selecting all samples from positive class.")
            n_positive = min(n, len(indices_predicted_positive))
            n_negative = 0
            
        else:
            # Normal case - calculate how many samples to select from each group based on split_ratio
            desired_n_positive = max(1, int(n * self.split_ratio))
            desired_n_negative = n - desired_n_positive
            
            # Check if we have enough samples in each class
            available_n_positive = len(indices_predicted_positive)
            available_n_negative = len(indices_predicted_negative)
            
            # If not enough positive samples, use as many as available and fill with negatives
            if available_n_positive < desired_n_positive:
                n_positive = available_n_positive
                n_negative = min(n - n_positive, available_n_negative)  # Try to fill with negatives
                
                # In the rare case where even combined there aren't enough samples
                if n_positive + n_negative < n:
                    print(f"Warning: Not enough samples in both classes combined. Requested {n}, but only have {n_positive + n_negative}")
                    
            # If not enough negative samples, use as many as available and fill with positives
            elif available_n_negative < desired_n_negative:
                n_negative = available_n_negative
                n_positive = min(n - n_negative, available_n_positive)  # Try to fill with positives
                
                # In the rare case where even combined there aren't enough samples
                if n_positive + n_negative < n:
                    print(f"Warning: Not enough samples in both classes combined. Requested {n}, but only have {n_positive + n_negative}")
                    
            # If we have enough samples in both classes, use the desired split
            else:
                n_positive = desired_n_positive
                n_negative = desired_n_negative
            
        print(f"Selecting {n_positive} from positives, {n_negative} from negatives")
        
        # Run discriminative active learning on each group
        indices_positive = self.dal_strategy.query(
            clf, dataset, indices_predicted_positive, indices_labeled, y, n=n_positive
        ) if n_positive > 0 else np.array([], dtype=indices_labeled.dtype)
        
        indices_negative = self.dal_strategy.query(
            clf, dataset, indices_predicted_negative, indices_labeled, y, n=n_negative
        ) if n_negative > 0 else np.array([], dtype=indices_labeled.dtype)
        
        # Combine the results
        indices = np.concatenate([indices_positive, indices_negative])
        
        return indices

    def __str__(self):
        return f'NIHDAL(classifier_factory={str(self.classifier_factory)}, ' \
               f'num_iterations={self.num_iterations}, unlabeled_factor={self.unlabeled_factor}, ' \
               f'split_ratio={self.split_ratio})'

# Functions for creating the data ------------------------------------------------------------

def load_and_format_dataset(train_test_split_ratio = 0.2, transformer_model_name = 'distilroberta-base', random_state=42):
    # Load data
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')   
    dat = dataset['train'].to_pandas()

    # Group by comment_id and text, and calculate aggregates
    dat = dat.groupby(['comment_id', 'text']).agg({
        'annotator_id': 'count',  # Count of records
        'hatespeech': 'mean',
        'hate_speech_score': 'mean',
        'target_religion': 'mean',
        'target_religion_atheist': 'mean',
        'target_religion_buddhist': 'mean',
        'target_religion_christian': 'mean',
        'target_religion_hindu': 'mean',
        'target_religion_jewish': 'mean',
        'target_religion_mormon': 'mean',
        'target_religion_muslim': 'mean',
        'target_religion_other': 'mean'
    }).reset_index()

    # Flatten the column names
    dat.columns = ['comment_id', 'text', 'n', 
                  'hatespeech', 'hate_speech_score',
                  'target_religion', 'target_religion_atheist', 'target_religion_buddhist',
                  'target_religion_christian', 'target_religion_hindu', 'target_religion_jewish',
                  'target_religion_mormon', 'target_religion_muslim', 'target_religion_other']

    # Create the label column
    dat['label'] = ((dat['hate_speech_score'] > 1) & (dat['target_religion'] == 1)).astype(int)

    # Create the strict label columns
    dat['target_religion_atheist_strict'] = (dat['target_religion_atheist'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_buddhist_strict'] = (dat['target_religion_buddhist'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_christian_strict'] = (dat['target_religion_christian'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_hindu_strict'] = (dat['target_religion_hindu'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_jewish_strict'] = (dat['target_religion_jewish'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_mormon_strict'] = (dat['target_religion_mormon'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_muslim_strict'] = (dat['target_religion_muslim'] == 1) & (dat['hate_speech_score'] > 1)
    dat['target_religion_other_strict'] = (dat['target_religion_other'] == 1) & (dat['hate_speech_score'] > 1)

    # Get list of all strict target columns
    strict_cols = [c for c in dat.columns if c.endswith('_strict')]

    # First create a column with all targeted religions
    dat['target_religions_list'] = ''
    for col in strict_cols:
        religion = col.replace('target_religion_', '').replace('_strict', '')
        dat.loc[dat[col] == True, 'target_religions_list'] = dat.loc[dat[col] == True, 'target_religions_list'] + religion + ','

    # Remove trailing comma
    dat['target_religions_list'] = dat['target_religions_list'].str.rstrip(',')

    # Count how many religions are targeted
    dat['target_religions_count'] = (dat[strict_cols].sum(axis=1)).astype(int)
    
    # Remove inconsistent rows where label is 1 but no specific religions are marked
    inconsistent_rows = (dat['label'] == 1) & (dat['target_religions_count'] == 0)
    if sum(inconsistent_rows) > 0:
        print(f"Removing {sum(inconsistent_rows)} rows where label=1 but no specific religions are targeted")
        dat = dat[~inconsistent_rows].reset_index(drop=True)

    # Get counts for each religion to determine which are smallest
    religion_counts = {}
    for col in strict_cols:
        religion = col.replace('target_religion_', '').replace('_strict', '')
        religion_counts[religion] = dat[col].sum()

    # Sort religions by count (smallest first)
    religions_by_size = sorted(religion_counts.keys(), key=lambda r: religion_counts[r])

    # Initialize strat_col as 'none'
    dat['strat_col'] = 'none'

    # For each row, find the smallest targeted religion
    for _, row in dat.iterrows():
        # Skip if no religion is targeted
        if row['target_religions_count'] == 0:
            continue
        
        # Find the smallest targeted religion for this row
        for religion in religions_by_size:
            col = f'target_religion_{religion}_strict'
            if row[col]:
                dat.loc[_, 'strat_col'] = religion
                break

    # Split the data (80% train, 20% test) with stratification by subclass
    # Create separate train/test splits for each strat_col value
    all_strat_values = dat['strat_col'].unique()
    train_indices = []
    test_indices = []
    
    # Print the distribution before split
    print("\nSubclass distribution before split:")
    for strat_val in all_strat_values:
        subset_size = sum(dat['strat_col'] == strat_val)
        subset_pct = subset_size / len(dat) * 100
        print(f"{strat_val}: {subset_size} samples ({subset_pct:.2f}%)")
    
    # Split each stratum separately to maintain distribution
    for strat_val in all_strat_values:
        # Get indices for this stratum
        stratum_indices = np.where(dat['strat_col'] == strat_val)[0]
            
        # Split this stratum
        stratum_train, stratum_test = train_test_split(
            stratum_indices,
            test_size=train_test_split_ratio,
            random_state=random_state
        )
        
        train_indices.extend(stratum_train)
        test_indices.extend(stratum_test)
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Print the distribution after split
    print("\nSubclass distribution after split:")
    print("Train set:")
    for strat_val in all_strat_values:
        train_count = sum(dat.iloc[train_indices]['strat_col'] == strat_val)
        train_pct = train_count / len(train_indices) * 100
        print(f"{strat_val}: {train_count} samples ({train_pct:.2f}%)")
    
    print("\nTest set:")
    for strat_val in all_strat_values:
        test_count = sum(dat.iloc[test_indices]['strat_col'] == strat_val)
        test_pct = test_count / len(test_indices) * 100
        print(f"{strat_val}: {test_count} samples ({test_pct:.2f}%)")
    
    # Create train and test dataframes
    train_df = dat.iloc[train_indices].reset_index(drop=True)
    test_df = dat.iloc[test_indices].reset_index(drop=True)
    
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

    # Create TransformersDataset for train and test
    train_dataset = TransformersDataset.from_arrays(
        train_df['text'],
        train_df['label'],
        tokenizer,
        max_length=100,
        target_labels=np.array([0, 1])
    )

    test_dataset = TransformersDataset.from_arrays(
        test_df['text'],
        test_df['label'],
        tokenizer,
        max_length=100,
        target_labels=np.array([0, 1])
    )

    return train_dataset, test_dataset, train_df, test_df

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

    # Set up transformer model
    transformer_model = TransformerModelArguments(transformer_model_name)

    # Set up classifier factory
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
        query_strategy = NIHDAL(classifier_factory=clf_factory, num_iterations=10)
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

def random_initialization_custom(dataset, dataset_df, n_samples=100, strategy='random', non_sample=None):
    """Randomly initializes data points based on different strategies.

    Parameters
    ----------
    dataset : TransformersDataset
        The dataset containing the features and labels for model training.
    dataset_df : pd.DataFrame
        The dataframe containing additional information for stratification.
    n_samples : int
        Number of samples to draw.
    strategy : str
        One of 'random', 'stratified', or 'biased'.
        - 'random': 50% negative, 50% positive examples
        - 'stratified': 50% negative, 50% positive, but stratified by strat_col
        - 'biased': 50% negative, 50% positive, but positives only from 'muslim' strat_col
    non_sample : list or np.ndarray
        Indices to exclude from sampling.

    Returns
    -------
    indices : np.ndarray
        Indices of the selected samples.
    """
    # Get labels as a numpy array
    y = dataset.y
    
    expected_samples_per_class = np.floor(n_samples/2).astype(int)
    
    if non_sample is None:
        non_sample = []
    
    if strategy == 'random':
        # Basic random sampling: 50% positive, 50% negative
        pos_indices = [i for i, lab in enumerate(y) if lab == 1 and i not in non_sample]
        neg_indices = [i for i, lab in enumerate(y) if lab == 0 and i not in non_sample]
        
        # Handle case where we don't have enough samples
        pos_samples = min(expected_samples_per_class, len(pos_indices))
        neg_samples = min(expected_samples_per_class, len(neg_indices))
        
        # Randomly sample
        pos_sample = random.sample(pos_indices, pos_samples)
        neg_sample = random.sample(neg_indices, neg_samples)
        
        return np.array(pos_sample + neg_sample)
    
    elif strategy == 'stratified':
        # Stratified sampling: 50% positive, 50% negative, maintaining strat_col distribution
        pos_indices = [i for i, lab in enumerate(y) if lab == 1 and i not in non_sample]
        neg_indices = [i for i, lab in enumerate(y) if lab == 0 and i not in non_sample]
        
        # Group positive indices by strat_col
        strat_groups = {}
        for i in pos_indices:
            strat_val = dataset_df.iloc[i]['strat_col']
            if strat_val not in strat_groups:
                strat_groups[strat_val] = []
            strat_groups[strat_val].append(i)
        
        # Calculate samples per strat_group proportionally
        total_pos = len(pos_indices)
        pos_sample = []
        
        for strat_val, indices in strat_groups.items():
            # Skip 'none' category if there are other categories
            if strat_val == 'none' and len(strat_groups) > 1:
                continue
                
            group_size = len(indices)
            group_samples = int(np.ceil((group_size / total_pos) * expected_samples_per_class))
            group_samples = min(group_samples, group_size)  # Don't sample more than available
            
            pos_sample.extend(random.sample(indices, group_samples))
        
        # If we sampled too many, subsample randomly
        if len(pos_sample) > expected_samples_per_class:
            pos_sample = random.sample(pos_sample, expected_samples_per_class)
        
        # Sample negatives
        neg_samples = min(expected_samples_per_class, len(neg_indices))
        neg_sample = random.sample(neg_indices, neg_samples)
        
        return np.array(pos_sample + neg_sample)
    
    elif strategy == 'biased':
        # Biased sampling: 50% negative, 50% positive but positives only from 'muslim' strat_col
        neg_indices = [i for i, lab in enumerate(y) if lab == 0 and i not in non_sample]
        
        # Get positive indices only from muslim strat_col
        muslim_pos_indices = [i for i, lab in enumerate(y) 
                             if lab == 1 and i not in non_sample 
                             and dataset_df.iloc[i]['strat_col'] == 'muslim']
        
        # If not enough muslim samples, fallback to random positives
        if len(muslim_pos_indices) < expected_samples_per_class:
            print(f"Warning: Not enough 'muslim' samples ({len(muslim_pos_indices)}). "
                  f"Adding other positive samples to reach {expected_samples_per_class}.")
            other_pos_indices = [i for i, lab in enumerate(y) 
                                if lab == 1 and i not in non_sample 
                                and dataset_df.iloc[i]['strat_col'] != 'muslim']
            
            # Sample all muslims
            pos_sample = muslim_pos_indices.copy()
            
            # Add other positives if needed
            remaining = expected_samples_per_class - len(pos_sample)
            if remaining > 0 and other_pos_indices:
                additional = random.sample(other_pos_indices, 
                                         min(remaining, len(other_pos_indices)))
                pos_sample.extend(additional)
        else:
            # We have enough muslim samples
            pos_sample = random.sample(muslim_pos_indices, expected_samples_per_class)
        
        # Sample negatives
        neg_samples = min(expected_samples_per_class, len(neg_indices))
        neg_sample = random.sample(neg_indices, neg_samples)
        
        return np.array(pos_sample + neg_sample)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Use one of: 'random', 'stratified', 'biased'.")

def evaluate(active_learner, train, test, train_df=None, test_df=None):

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

    # Add subgroup metrics if dataframes are provided
    if train_df is not None and test_df is not None:
        # Get all religious subgroup columns
        religion_cols = [col for col in test_df.columns if col.startswith('target_religion_') and col.endswith('_strict')]
        
        # Save the subgroup data
        r['train_religion_subgroups'] = train_df[religion_cols]
        r['test_religion_subgroups'] = test_df[religion_cols]
        
        # Calculate metrics for each subgroup
        for col in religion_cols:
            religion = col.replace('target_religion_', '').replace('_strict', '')
            
            # Test set metrics for this subgroup
            if sum(test_df[col]) > 0:  # Skip if no examples in this subgroup
                # Filter to just this subgroup
                subgroup_indices = test_df[col].values.astype(bool)
                subgroup_y_true = test.y[subgroup_indices]
                subgroup_y_pred = y_pred_test[subgroup_indices]
                
                # Calculate metrics
                r[f'Test accuracy_{religion}'] = accuracy_score(subgroup_y_pred, subgroup_y_true)
                r[f'Test F1_{religion}'] = f1_score(subgroup_y_pred, subgroup_y_true)
                r[f'Test precision_{religion}'] = precision_score(subgroup_y_pred, subgroup_y_true)
            
            # Train set metrics for this subgroup
            if sum(train_df[col]) > 0:  # Skip if no examples in this subgroup
                # Filter to just this subgroup
                subgroup_indices = train_df[col].values.astype(bool)
                subgroup_y_true = train.y[subgroup_indices]
                subgroup_y_pred = y_pred[subgroup_indices]
                
                # Calculate metrics
                r[f'Train accuracy_{religion}'] = accuracy_score(subgroup_y_pred, subgroup_y_true)
                r[f'Train F1_{religion}'] = f1_score(subgroup_y_pred, subgroup_y_true)
                r[f'Train precision_{religion}'] = precision_score(subgroup_y_pred, subgroup_y_true)

    print('Test accuracy:', r['Test accuracy'], 'Test F1:', r['Test F1'])

    return r

# Old functions ------------------------------------------------------------

def active_learning_loop(active_learner, train, test, train_df, test_df, num_queries, strategy='random'):

    # Initialize with first sample
    indices_labeled = initialize_active_learner(active_learner, train, train_df, strategy=strategy)

    results = []
    results.append(evaluate(active_learner, train[indices_labeled], test, train_df.iloc[indices_labeled], test_df))

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
        res = evaluate(active_learner, train[indices_labeled], test, train_df.iloc[indices_labeled], test_df)

        # Track the counts directly in res
        res['counts'] = {}
            
        res['counts']['all'] = {
            'selected': len(indices_queried),
            'target': int(sum(y)),
        }
        
        # Track selection by subgroup
        religion_cols = [col for col in train_df.columns if col.startswith('target_religion_') and col.endswith('_strict')]
        for col in religion_cols:
            religion = col.replace('target_religion_', '').replace('_strict', '')
            subgroup_indices = train_df.iloc[indices_queried][col].values.astype(bool)
            res['counts'][religion] = {
                'selected': int(sum(subgroup_indices)),
                'target': int(sum(y[subgroup_indices])) if sum(subgroup_indices) > 0 else 0
            }

        print(res['counts'])

        results.append(res)

    return results

def initialize_active_learner(active_learner, dataset, dataset_df, strategy='random'):
    """Initialize the active learner with initial data points.
    
    Parameters
    ----------
    active_learner : PoolBasedActiveLearner
        The active learning model to initialize
    dataset : TransformersDataset
        The dataset containing features and labels
    dataset_df : pd.DataFrame
        The dataframe with additional information for stratification
    strategy : str
        Initialization strategy ('random', 'stratified', or 'biased')
    
    Returns
    -------
    indices_initial : np.ndarray
        The indices of the initial samples
    """
    # Simulate an initial labeling to warm-start the active learning process
    indices_initial = random_initialization_custom(dataset=dataset, dataset_df=dataset_df, 
                                                 n_samples=100, strategy=strategy)

    active_learner.initialize(indices_initial, dataset.y[indices_initial])

    return indices_initial

# Main body -----------------------------------------------------------

if __name__ == '__main__':

    datasets.logging.set_verbosity_error()
    datasets.logging.get_verbosity = lambda: logging.NOTSET

    transformer_model_name = 'distilroberta-base'
    output_dir = '/n/netscratch/economics/Lab/esilcock/nihdal_results/hate_speech_sim0427'
    num_queries = 3

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process each seed once
    for seed in [42]:  # 42, 12731, 65372, 97, 163
        print(f'#################{seed}##################')
        
        # Set seeds for everything
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load dataset once for this seed
        train, test, train_df, test_df = load_and_format_dataset(
            train_test_split_ratio = 0.2,
            transformer_model_name = transformer_model_name,
            random_state=seed
        )
        
        # Generate initial indices once for this seed
        indices_initial = random_initialization_custom(
            dataset=train, 
            dataset_df=train_df, 
            n_samples=100, 
            strategy='random'
        )
        
        # Now run different active learning methods with the same initial data
        # for als in ['NIHDAL', 'DAL2', 'Core Set', 'Least Confidence', 'Random']:
        for als in ['Random']:
            print(f'****************{als}**********************')

            # Reset seeds to ensure all random operations are consistent
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            selected_descr = None
            
            # Set up the active learner for this method
            active_learner = set_up_active_learner(transformer_model_name, active_learning_method=als, train_dataset=train)
            
            # Directly initialize with the pre-generated indices instead of calling initialize_active_learner
            active_learner.initialize(indices_initial, train.y[indices_initial])
            
            # Modified active learning loop that skips initialization
            results = []
            # Add initial evaluation
            results.append(evaluate(active_learner, train[indices_initial], test, train_df.iloc[indices_initial], test_df))
            
            # Run active learning queries
            indices_labeled = indices_initial.copy()
            for i in range(num_queries):  
                # Query samples to label
                indices_queried = active_learner.query(num_samples=100)
                
                # Simulate labelling
                y = train.y[indices_queried]
                
                # Return the labels for the current query to the active learner
                active_learner.update(y)
                
                indices_labeled = np.concatenate([indices_queried, indices_labeled])
                
                print('---------------')
                print(f'Iteration #{i} ({len(indices_labeled)} samples)')
                res = evaluate(active_learner, train[indices_labeled], test, train_df.iloc[indices_labeled], test_df)
                
                # Track the counts directly in res
                res['counts'] = {}
                
                res['counts']['all'] = {
                    'selected': len(indices_queried),
                    'target': int(sum(y)),
                }
                
                # Track selection by subgroup
                religion_cols = [col for col in train_df.columns if col.startswith('target_religion_') and col.endswith('_strict')]
                for col in religion_cols:
                    religion = col.replace('target_religion_', '').replace('_strict', '')
                    subgroup_indices = train_df.iloc[indices_queried][col].values.astype(bool)
                    res['counts'][religion] = {
                        'selected': int(sum(subgroup_indices)),
                        'target': int(sum(y[subgroup_indices])) if sum(subgroup_indices) > 0 else 0
                    }
                
                print(res['counts'])
                results.append(res)
            
            # Save results for this method
            with open(f'{output_dir}/hate_speech_{als}_results_{seed}_unbiased.pkl', 'wb') as f:
                pickle.dump(results, f)


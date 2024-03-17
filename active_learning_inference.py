import os
import json
from tqdm import tqdm

os.environ['TRANSFORMERS_CACHE'] = '.cache/'
import numpy as np
from transformers import AutoTokenizer

import small_text
from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    DiscriminativeActiveLearning,
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

        # Predict target or other for unlabelled
        # preds = last_stable_model.predict(train)
        preds = active_learner.classifier.predict(train)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        target_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 1])
        other_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 0])

        # Describe
        print(f'There are {len(target_indices_unlabeled)} predicted target examples')
        print(f'There are {len(other_indices_unlabeled)} predicted non-target examples')
        print(f'There are {len(target_indices_labeled)} labelled target examples')
        print(f'There are {len(other_indices_labeled)} labelled non-target examples')

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

        return selected_indices


class NIHDAL_2(DiscriminativeActiveLearning_amended):

    """Similar to Discriminative Active Learning, but applied reweighting the pool.
    """

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):

        # Predict target or other for unlabelled data
        preds = active_learner.classifier.predict(train)

        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        # Describe
        print(f'There are {len(target_indices_unlabeled)} predicted target examples')
        print(f'There are {len(other_indices_unlabeled)} predicted non-target examples')

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

        return selected_indices


def find_sep_token(tokenizer):

    """
    Returns sep token for given tokenizer
    """

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


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


transformer_model_name = 'roberta-large'
als = 'NIHDAL'

publications = [
    'The Sun (England)',
    'thesun.co.uk',
    'Daily Star',
    'Daily Star Online',
    'Daily Star Sunday',
    'The Daily Mail and Mail on Sunday (London)',
    'mirror.co.uk',
    'Daily Mirror',
    'The Express',
    'The Sunday Express',
    'The News of the World',
    'The Evening Standard (London)',
    'standard.co.uk',
    'The People',
    'Metro (UK)',
    'City A.M.',
    'Cityam.com',
    'The Times (London)',
    'The Sunday Times (London)',
    'thetimes.co.uk',
    'The Daily Telegraph (London)',
    'The Daily Telegraph Online',
    'The Sunday Telegraph (London)',
    'The Guardian (London)',
    'The Observer (London)',
    'i - Independent Print Ltd',
    'The Independent (United Kingdom)',
    'Liverpool Post',
    'liverpoolecho.co.uk',
    'Liverpool Echo',
]

# Open all data
sample_list = []

for publication in tqdm(publications):

    publication_fn = publication.replace(' ', '_')

    with open(f"Sun_data/{publication_fn}/cleaned_sample_data.json") as f:
        clean_dat = json.load(f)

    with open(f"Sun_data/sample_indices_{publication_fn}.json") as f:
        sample = json.load(f)

    # Take sample
    for s in sample:
        try:
            sample_list.append(clean_dat[str(s)])
        except:
            pass

# Labelled data 
with open('Labelled_data/harmonised_sample_1.json') as f:
    labelled_data = json.load(f)

parsed_labelled_data = {}

for task in labelled_data:
    lab = task['annotations'][0]['result'][0]['value']['choices'][0]
    ln_id = task['data']['ln_id']
    parsed_labelled_data[ln_id] = lab

texts = []
currently_labelled = []
labels = []

tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
sep = find_sep_token(tokenizer)
print(f"Sep: {sep}")

for idx, article in tqdm(enumerate(sample_list)):

    # Check and add to labels
    if article['ln_id'] in parsed_labelled_data:
        currently_labelled.append(idx)

        lab = parsed_labelled_data[article['ln_id']]
        if lab == 'Irrelevant':
            labels.append(0)
        else:
            labels.append(1)

    # Create pool 
    text = str(article['headline']) + sep + str(article['article'])
    texts.append(text)

print(f"Pool size: {len(texts)}")
print(f"of which {len(labels)} are labelled")
      
assert len(labels) == len(parsed_labelled_data)
currently_labelled = np.array(currently_labelled)
labels = np.array(labels)

unlabels = [small_text.base.LABEL_UNLABELED for i in range(len(texts))]

lab_array = np.arange(2)

train = TransformersDataset.from_arrays(
    texts,
    unlabels,
    tokenizer,
    max_length=100,
    target_labels=lab_array
)

## Active Learning
active_learner = set_up_active_learner(transformer_model_name, active_learning_method=als)

print(type(currently_labelled))
print(len(currently_labelled))
print(type(labels))
print(len(labels))

active_learner.initialize_data(currently_labelled, labels)

indices_queried = active_learner.query(num_samples=100)

# Format for label studio 

print(indices_queried)
# Heavily based on https://github.com/webis-de/small-text/blob/main/examples/notebooks/01-active-learning-for-text-classification-with-small-text-intro.ipynb

# Also see:
# *   For multi-class classification: https://github.com/webis-de/small-text/blob/main/examples/examplecode/transformers_multiclass_classification.py, https://github.com/webis-de/small-text/blob/main/examples/examplecode/transformers_multilabel_classification.py
# *   Stopping criteria: https://github.com/webis-de/small-text/blob/main/examples/notebooks/02-active-learning-with-stopping-criteria.ipynb

import datasets
import torch
import numpy as np
from transformers import AutoTokenizer
import small_text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import random 
from copy import deepcopy


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


def sample_and_tokenize_data(dataset_name, tokenization_model, target_labels=[0], biased = False):

    raw_dataset = datasets.load_dataset(dataset_name)
    dataset_to_change = datasets.load_dataset(dataset_name)

    dataset_to_change['train'] = make_binary(raw_dataset['train'], target_labels=target_labels)
    dataset_to_change['test'] = make_binary(raw_dataset['test'], target_labels=target_labels)

    # if biased:
    #     unsampled_train_indices = [i for i, lab in enumerate(raw_dataset['train']['label']) if lab == target_labels[1]]

    #     dataset_to_change['train'], bias_indices = make_imbalanced(dataset_to_change['train'], indices_to_track=unsampled_train_indices)

    # else:
    #     dataset_to_change['train'] = make_imbalanced(dataset_to_change['train'])

    # dataset_to_change['test'] = make_imbalanced(dataset_to_change['test'])

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    # Covert to a TransformersDataset
    labels = np.arange(2)

    train_dat = small_text.TransformersDataset.from_arrays(
        texts=dataset_to_change['train']['text'],
        y=dataset_to_change['train']['label'],
        tokenizer=tokenizer,
        max_length=100,
        target_labels=labels
        )

    test_dat = small_text.TransformersDataset.from_arrays(
        texts=dataset_to_change['test']['text'],
        y=dataset_to_change['test']['label'],
        tokenizer=tokenizer,
        max_length=100,
        target_labels=labels
        )

    assert not train_dat.multi_label

    if biased:
        return train_dat, test_dat, bias_indices

    else:
        return train_dat, test_dat


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


class DiscriminativeActiveLearning_amended(small_text.query_strategies.strategies.DiscriminativeActiveLearning):

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
#class NIHDAL(small_text.query_strategies.strategies.DiscriminativeActiveLearning):

    """Similar to Discriminative Active Learning, but applied on the predicted positives and 
     negatives separately. 
    """

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        # Predict target or other for unlabelled data
        preds = last_stable_model.predict(train)
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


def set_up_active_learner(train_dat, classification_model, active_learning_method):


    # Set up classifier
    transformer_model = small_text.TransformerModelArguments(classification_model)

    # Hyperparams - options here: https://small-text.readthedocs.io/en/v1.3.3/api/classifier.html#small_text.integrations.transformers.classifiers.TransformerBasedClassification.__init__

    model_selection = small_text.training.ModelSelection(
        default_select_by='val_f1',
        metrics=[
            small_text.training.metrics.Metric('val_f1', lower_is_better=False),
            small_text.training.metrics.Metric('val_loss', lower_is_better=True), 
            small_text.training.metrics.Metric('val_acc', lower_is_better=False), 
            small_text.training.metrics.Metric('train_loss', lower_is_better=True), 
            small_text.training.metrics.Metric('train_acc', lower_is_better=False)
        ],
        required=['val_f1']
    )

    hyperparameters = {
        'device': 'cuda',
        'mini_batch_size': 50,
        'num_epochs': 20, # default=10
        'class_weight': 'balanced', #  If ‘balanced’, then the loss function is weighted inversely proportional to the label distribution to the current train set. DAL Blog post said this was important
        'lr': 2e-5,  # default=2e-5, AL for BERT 5e-05
        # 'validation_set_size': 0.1, # default=0.1
        # validations_per_epoch: 1 # default=1,
        'model_selection': model_selection
        }

    classifier = small_text.TransformerBasedClassificationFactory(
        transformer_model,
        num_classes=2,
        kwargs=dict(hyperparameters)
        )

    if active_learning_method in ["DAL", "NIHDAL"]:

        classifier_b = small_text.TransformerBasedClassificationFactory(
            transformer_model,
            num_classes=2,
            kwargs=dict(hyperparameters)
            )


        if active_learning_method == "DAL":
            query_strategy = small_text.query_strategies.strategies.DiscriminativeActiveLearning(
            # query_strategy = DiscriminativeActiveLearning_amended(
                classifier_b,
                num_iterations=10, # This is referred to as the number of sub-batches in the DAL paper - they found 10-20 worked well. We might want to increase this
                unlabeled_factor=10,  # This means the unlabelled gets downsampled so it's only ever 10x the size of the labelled
                pbar='tqdm'
                )
        else:
            query_strategy = NIHDAL(
                classifier_b,
                num_iterations=10, # This is referred to as the number of sub-batches in the DAL paper - they found 10-20 worked well. We might want to increase this
                unlabeled_factor=10,  # This means the unlabelled gets downsampled so it's only ever 10x the size of the labelled
                pbar='tqdm'
                )

        ## DAL paper they find that early stopping is important and they use 0.98 on accuracy
        early_stopping = small_text.training.early_stopping.EarlyStopping(small_text.training.metrics.Metric('train_acc', lower_is_better=False), threshold=0.98)

        # Put it all together
        a_learner = small_text.PoolBasedActiveLearner(
            classifier,
            query_strategy,
            train_dat,
            reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining.
            fit_kwargs={'early_stopping': early_stopping}
            )

        return a_learner

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

    # Put it all together
    a_learner = small_text.PoolBasedActiveLearner(
        classifier,
        query_strategy,
        train_dat,
        reuse_model=False, # Reuses the previous model during retraining (if a previous model exists), otherwise creates a new model for each retraining.
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


if __name__ == '__main__':

    ## Fix seeds
    SEED = 42 #12731 # 65372 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Choose sampling
    BIASED = False

    ## Choose backbone
    # transformer_model_name = 'bert-base-uncased'
    transformer_model_name = 'distilroberta-base'
    # found pretty impressive performance with 'paraphrase-MiniLM-L3-v2' https://github.com/webis-de/small-text/blob/main/examples/notebooks/03-active-learning-with-setfit.ipynb
    # roberta-base
    # roberta-large
    # deberta? Never really had great results on this

    ## Sample data
    if BIASED:
        train, test, unsampled_indices = sample_and_tokenize_data(
            dataset_name='ag_news',
            target_labels=[0, 2],
            tokenization_model = transformer_model_name,
            biased=True
        )

    else:
        train, test = sample_and_tokenize_data(
            dataset_name='ag_news',  # d
            # dataset_name = 'rotten_tomatoes', # movie reviews, labelled as either positive or negative
            # go-emotions dataset (27 emotions + 1 neutral class)
            target_labels=[0],
            tokenization_model = transformer_model_name,
            biased=False
        )

    # for als in ["Random", "Least Confidence", "BALD", "BADGE", "DAL", "Core Set", "Contrastive", "NIHDAL", "Expected Gradient Length"]:
    for als in ["DAL"]:

        print(f"***************************{als}******************************")

        ## Take first labelled sample
        # Simulates an initial labeling to warm-start the active learning process

        if BIASED:
            # Only sample from one side of the target class 
            indices_labeled = random_initialization_biased(train.y, n_samples=100, non_sample=unsampled_indices)

            lt_1 = len([i for i in indices_labeled if train.y[i] == 1 and i not in unsampled_indices])
            lt_2 = len([i for i in indices_labeled if train.y[i] == 1 and i in unsampled_indices])
            lt=lt_1 + lt_2
            lo = len([i for i in indices_labeled if train.y[i] == 0])
            print(f'Selected {lt_1} samples of target class a), {lt_2} of target class b) {lo} of non-target class for labelling')

            

        else:
            # # Random
            # indices_labeled = small_text.random_initialization(train.y, n_samples=100)
            # Random, stratified by label
            indices_labeled = small_text.random_initialization_balanced(train.y, n_samples=100)
            lt = len([i for i in indices_labeled if train.y[i] == 1])
            lo = len([i for i in indices_labeled if train.y[i] == 0])
            print(f'Selected {lt} samples of target class and {lo} of non-target class for labelling')

        ## Set up active learner
        active_learner = set_up_active_learner(
            train,
            classification_model=transformer_model_name,
            # active_learning_method="DAL"
            # active_learning_method="Least Confidence"
            # active_learning_method="Prediction Entropy"
            active_learning_method=als
        )

        # Initalise - trains first pass of classification model
        active_learner.initialize_data(indices_labeled, train.y[indices_labeled])

        ## Query
        NUM_QUERIES = 10

        results = []
        res = evaluate(active_learner, train[indices_labeled], test)

        if BIASED:
            res['on_first_topic'] = lt_1
            res['on_second_topic'] = lt_2
            res['not_on_topic'] = lo
        else:
            res['on_topic'] = lt
            res['not_on_topic'] = lo
        results.append(res)

        ## Keep track of last stable model 
        last_stable_model = deepcopy(active_learner.classifier)

        for i in range(NUM_QUERIES):

            # Query 100 samples
            indices_queried = active_learner.query(num_samples=100)

            # Label these (simulated)
            y = train.y[indices_queried]

            if BIASED:
                lt_1 = len([i for i in range(len(y)) if y[i] == 1 and i not in unsampled_indices])
                lt_2 = len([i for i in range(len(y)) if y[i] == 1 and i in unsampled_indices])
                lt=lt_1 + lt_2
                lo = len([i for i in y if i == 0])
                print(f'Selected {lt_1} samples of target class a), {lt_2} of target class b) {lo} of non-target class for labelling')

            else:
                lt = len([i for i in y if i == 1])
                lo = len([i for i in y if i == 0])
                print(f'Selected {lt} samples of target class and {lo} of non-target class for labelling')

            indices_labeled = np.concatenate([indices_queried, indices_labeled])

            # Perform an update step, which passes the label for each of the queried indices
            # At end of the update step the classification model is retrained using all available labels
            active_learner.update(y)

            print('---------------')
            print(f'Iteration #{i} ({len(indices_labeled)} samples total)')

            # Evaluate classification model
            res = evaluate(active_learner, train[indices_labeled], test)

            if BIASED:
                res['on_first_topic'] = lt_1
                res['on_second_topic'] = lt_2
                res['not_on_topic'] = lo

            else:
                res['on_topic'] = lt
                res['not_on_topic'] = lo
            
            results.append(res)

            # Update last stable model - used for DAL and NIHDAL
            if res['Test F1'] > 0:
                print('Classification model updated')
                last_stable_model = deepcopy(active_learner.classifier)

            else:
                print('Classification model not updated')


        if BIASED:
            with open(f'{als}_results_{SEED}_biased.json', 'w') as f:
                json.dump(results, f, indent=4)

        else:
            with open(f'{als}_results_{SEED}.json', 'w') as f:
                json.dump(results, f, indent=4)


    # Todo:
            # Change hyperparameters on classification model so you're not getting odd behaviour
            # Decide on other measures of success
            # Try with other data
            # Try repeating a few times

    # Examples 
                
                # 27 - this is meant to be business
                # {'text': 'AUDIT CONFIRMS CHAVEZ VICTORY An audit of last week #39;s recall vote in Venezuela has found no evidence of fraud in the process that endorsed President Hugo Chavez as leader.', 'label': 1}
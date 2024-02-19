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


def make_binary(dataset, target_label):

    # Create mapping
    num_classes = dataset.features['label'].num_classes

    class_mapping = {lab: 0 for lab in range(num_classes)}
    class_mapping[target_label] = 1

    # Apply the mapping to change the labels
    binary_dataset = dataset.map(lambda example: {'label': class_mapping[example['label']]})

    # Update metadata
    new_features = datasets.Features({
        'text': binary_dataset.features['text'],
        'label': datasets.ClassLabel(names = ['merged', 'target'], num_classes=2)
        })
    binary_dataset = binary_dataset.cast(new_features)

    return binary_dataset


def make_imbalanced(dataset):

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

    return imbalanced_dataset


def sample_and_tokenize_data(dataset_name, tokenization_model, target_label=0):


    raw_dataset = datasets.load_dataset(dataset_name)

    raw_dataset['train'] = make_binary(raw_dataset['train'], target_label=target_label)
    raw_dataset['test'] = make_binary(raw_dataset['test'], target_label=target_label)

    raw_dataset['train'] = make_imbalanced(raw_dataset['train'])
    raw_dataset['test'] = make_imbalanced(raw_dataset['test'])

    ## Prep dataset
    num_classes = raw_dataset['train'].features['label'].num_classes

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    # Covert to a TransformersDataset
    target_labels = np.arange(num_classes)

    train_dat = small_text.TransformersDataset.from_arrays(
        texts=raw_dataset['train']['text'],
        y=raw_dataset['train']['label'],
        tokenizer=tokenizer,
        max_length=100,
        target_labels=target_labels
        )

    test_dat = small_text.TransformersDataset.from_arrays(
        texts=raw_dataset['test']['text'],
        y=raw_dataset['test']['label'],
        tokenizer=tokenizer,
        max_length=100,
        target_labels=target_labels
        )
    
    assert not train_dat.multi_label
    
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

        clf = active_learner._clf

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
        preds = active_learner.classifier.predict(train)
        target_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 1])
        other_indices_unlabeled = np.array([i for i in indices_unlabeled if preds[i] == 0])

        # If there are no predicted targets 
        if len(other_indices_labeled) == 0 or len(other_indices_unlabeled):
            print("Classification model predicted all items with same label, reverting to DAL")

            indices = self.discriminative_active_learning(
                    dataset,
                    indices_unlabeled,
                    indices_labeled,
                    query_sizes
                )
            
            return indices


        else:
            query_sizes = self._get_query_sizes(self.num_iterations, int(n/2))

            target_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 1])
            other_indices_labeled = np.array([i for i in indices_labeled if train.y[i] == 0])

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

            return np.concatenate((target_indices, other_indices))


def set_up_active_learner(train_dat, classification_model, active_learning_method):


    # Set up classifier
    transformer_model = small_text.TransformerModelArguments(classification_model)

    # Hyperparams - options here: https://small-text.readthedocs.io/en/v1.3.3/api/classifier.html#small_text.integrations.transformers.classifiers.TransformerBasedClassification.__init__
    hyperparameters = {
        'device': 'cuda',
        'mini_batch_size': 50,
        'num_epochs': 5, # default=10
        'class_weight': 'balanced', #  If ‘balanced’, then the loss function is weighted inversely proportional to the label distribution to the current train set. DAL Blog post said this was important
        'lr': 5e-5,  # default=2e-5
        # 'validation_set_size': 0.1, # default=0.1
        # validations_per_epoch: 1 # default=1
        }

    classifier = small_text.TransformerBasedClassificationFactory(
        transformer_model,
        num_classes=2,
        kwargs=dict(hyperparameters)
        )

    if active_learning_method in ["DAL", "NIHDAL"]:

        if active_learning_method == "DAL":
            # query_strategy = small_text.query_strategies.strategies.DiscriminativeActiveLearning(
            query_strategy = DiscriminativeActiveLearning_amended(
                classifier,
                num_iterations=10, # This is referred to as the number of sub-batches in the DAL paper - they found 10-20 worked well. We might want to increase this
                unlabeled_factor=10,  # This means the unlabelled gets downsampled so it's only ever 10x the size of the labelled
                pbar='tqdm'
                )
        else:
            query_strategy = NIHDAL(
                classifier,
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
        query_strategy = small_text.integrations.pytorch.query_strategies.strategies.ExpectedGradientLength(num_classes=2)
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


if __name__ == '__main__':

    ## Fix seeds
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ## Choose backbone
    # transformer_model_name = 'bert-base-uncased'
    transformer_model_name = 'distilroberta-base'
    # found pretty impressive performance with 'paraphrase-MiniLM-L3-v2' https://github.com/webis-de/small-text/blob/main/examples/notebooks/03-active-learning-with-setfit.ipynb
    # roberta-base
    # roberta-large
    # deberta? Never really had great results on this

    ## Sample data
    train, test = sample_and_tokenize_data(
        dataset_name='ag_news',  # News data, labelled as 'World', 'Sports', 'Business', 'Sci/Tech'
        # dataset_name = 'rotten_tomatoes', # movie reviews, labelled as either positive or negative
        # go-emotions dataset (27 emotions + 1 neutral class)
        target_label=0,
        tokenization_model = transformer_model_name
    )

    # for als in ["Random", "Least Confidence", "BALD", "Expected Gradient Length", "BADGE", "DAL", "Core Set", "Contrastive", "NIHDAL"]:
    for als in ["NIHDAL"]:

        print(f"***************************{als}******************************")

        ## Take first labelled sample
        # Simulates an initial labeling to warm-start the active learning process

        # # Random
        # indices_labeled = small_text.random_initialization(train.y, n_samples=100)
        # Random, stratified by label
        indices_labeled = small_text.random_initialization_balanced(train.y, n_samples=100)
        lt = len([train.y[i] for i in indices_labeled if train.y[i] == 1])
        lo = len([train.y[i] for i in indices_labeled if train.y[i] == 0])
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
        res['on_topic'] = lt
        res['not_on_topic'] = lo
        results.append(res)

        for i in range(NUM_QUERIES):

            # Query 100 samples
            indices_queried = active_learner.query(num_samples=100)

            # Label these (simulated)
            y = train.y[indices_queried]

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
            res['on_topic'] = lt
            res['not_on_topic'] = lo
            results.append(res)

        with open(f'{als}_results.json', 'w') as f:
            json.dump(results, f, indent=4)

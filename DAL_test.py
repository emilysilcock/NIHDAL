import datasets
import logging
import torch
import numpy as np
from transformers import AutoTokenizer
from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    PredictionEntropy,
    DiscriminativeActiveLearning,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    random_initialization_balanced
)
from sklearn.metrics import accuracy_score, f1_score



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


### Preparation
datasets.logging.set_verbosity_error()

# disables the progress bar for notebooks: https://github.com/huggingface/datasets/issues/2651
datasets.logging.get_verbosity = lambda: logging.NOTSET

seed = 2022
torch.manual_seed(seed)
np.random.seed(seed)

## II. Data
raw_dataset = datasets.load_dataset('ag_news')

### Make binary 
raw_dataset['train'] = make_binary(raw_dataset['train'], target_labels=[0])
raw_dataset['test'] = make_binary(raw_dataset['test'], target_labels=[0])


num_classes = raw_dataset['train'].features['label'].num_classes

### Preparing the Data
transformer_model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(
    transformer_model_name
)

target_labels = np.arange(num_classes)

train = TransformersDataset.from_arrays(raw_dataset['train']['text'],
                                        raw_dataset['train']['label'],
                                        tokenizer,
                                        max_length=60,
                                        target_labels=target_labels)
test = TransformersDataset.from_arrays(raw_dataset['test']['text'],
                                       raw_dataset['test']['label'],
                                       tokenizer,
                                       max_length=60,
                                       target_labels=target_labels)

## III. Setting up the Active Learner


# simulates an initial labeling to warm-start the active learning process
def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_balanced(y_train, n_samples=20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


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

# query_strategy = PredictionEntropy()
query_strategy = DiscriminativeActiveLearning(classifier_factory=clf_factory_2, num_iterations=10)

active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
indices_labeled = initialize_active_learner(active_learner, train.y)

### Active Learning Loop
num_queries = 10


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = accuracy_score(y_pred_test, test.y)

    print('Train accuracy: {:.2f}'.format(accuracy_score(y_pred, train.y)))
    print('Test accuracy: {:.2f}'.format(test_acc))
    print('Train F1: {:.2f}'.format(f1_score(y_pred, train.y, average="weighted")))      ############################
    print('Test F1: {:.2f}'.format(f1_score(y_pred_test, test.y, average="weighted")))   ############################

    return test_acc


results = []
results.append(evaluate(active_learner, train[indices_labeled], test))


for i in range(num_queries):
    # ...where each iteration consists of labelling 20 samples
    indices_queried = active_learner.query(num_samples=20)

    # Simulate user interaction here. Replace this for real-world usage.
    y = train.y[indices_queried]

    # Return the labels for the current query to the active learner.
    active_learner.update(y)

    indices_labeled = np.concatenate([indices_queried, indices_labeled])

    print('---------------')
    print(f'Iteration #{i} ({len(indices_labeled)} samples)')
    results.append(evaluate(active_learner, train[indices_labeled], test))

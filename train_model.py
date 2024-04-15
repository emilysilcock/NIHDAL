from glob import glob
import numpy as np
from datetime import datetime
import json
import pandas as pd
import os
from tqdm import tqdm
import random 

import sklearn
import sklearn.model_selection

from datasets import load_dataset, load_metric, Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import wandb

os.environ['TRANSFORMERS_CACHE'] = '.cache/'


def find_sep_token(tokenizer):

    """
    Returns sep token for given tokenizer
    """

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


def open_and_reformat_ls_data(list_of_paths, label_dict, model):

    """
    Takes a list of label studio output files and returns a dataframe with article + label.
    Headlines, bylines and article texts are in "article", separated by the appropriate sep token for the specified
    model.
    """

    # Load all data
    labelled_data = []
    for path in list_of_paths:
        with open(path) as f:
            labelled_data.extend(json.load(f))

    # Reformat data
    if model == "No model":
        sep = " "
    else:
        sep = find_sep_token(tokenizer=AutoTokenizer.from_pretrained(model))

    ids = []
    texts = []
    labels = []
    positives = 0

    for data in labelled_data:
        label_text = data['annotations'][0]['result'][0]['value']['choices'][0]

        formatted_text = str(data['data']['headline']) + sep + str(data['data']['article'])

        try:
            texts.append(formatted_text)
            labels.append(label_dict[label_text])
            ids.append(data['data']['ln_id'])

        except:
            print(formatted_text)
            print(data['data']['ln_id'])

        if label_dict[label_text] == 1:
            positives += 1

    print(f"{len(labels)} labelled examples of which {positives} are positive")

    pd_data = pd.DataFrame(
        {
            'id': ids,
            'article': texts,
            'label': labels,
        }
    )

    return pd_data


def train_test_dev_split(ls_data_paths, label_dict, save_dir, model, test_perc=0.15, eval_perc=0.15, train_only_data_paths=[], split_by_scan=False, multi_lab=False):

    """
    Takes a list of label studio datapaths, reformats according to open_and_reformat_ls_data().
    Splits into train, eval and test splits.
    Saves splits in save_dir.
    """

    pd_data = open_and_reformat_ls_data(ls_data_paths, label_dict, model)

    # Split into test and train sets
    test_size = int(test_perc * len(pd_data))
    eval_size = int(eval_perc * len(pd_data))

    train_eval, test = sklearn.model_selection.train_test_split(pd_data, test_size=test_size, random_state=22)
    train, eval = sklearn.model_selection.train_test_split(train_eval, test_size=eval_size, random_state=17)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    train.to_csv(f'{save_dir}/train.csv', encoding='utf-8', index=False)
    eval.to_csv(f'{save_dir}/eval.csv', encoding='utf-8', index=False)
    test.to_csv(f'{save_dir}/test.csv', encoding='utf-8', index=False)

    print(len(train), "training examples")
    print(len(eval), "dev examples")
    print(len(test), "test examples")


def tokenize_data_for_finetuning(directory, hf_model, max_token_length):

    # Load data
    dataset = load_dataset('csv', data_files={'data': directory})

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset["article"], padding="max_length", truncation=True, max_length=max_token_length)



    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_data = tokenized_dataset["data"]

    return tokenized_data


def compute_metrics(eval_pred):
    metric0 = load_metric("accuracy")
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")

    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    accuracy = metric0.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels)["f1"]

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(
        train_dataset,
        eval_dataset,
        hf_model,
        eval_steps,
        batch_size,
        lr,
        epochs,
        save_dir,
        num_labels=2
):

    wandb.init(project="topic", entity="emilys")

    model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=num_labels)

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        output_dir=save_dir,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="wandb"
    )

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_path = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric

    return best_model_path, best_metric


def train_wrapper():

    run = wandb.init()
    name = f'rl_{wandb.config.batch_size}_{wandb.config.epochs}_{wandb.config.lr}_{wandb.config.max_len}'

    # Tokenize data
    datasets = {}
    for dataset in ["train", "eval", "test"]:
        datasets[dataset] = tokenize_data_for_finetuning(
            directory=f"/mnt/data01/AL/final_labelled_data/{dataset}.csv",
            hf_model=pretrained_model,
            max_token_length=wandb.config.max_len
        )

    # Train model
    train(
        datasets["train"],
        datasets["eval"],
        hf_model=pretrained_model,
        num_labels=2, 
        eval_steps=10,
        batch_size=wandb.config.batch_size,
        lr=wandb.config.lr,
        epochs=wandb.config.epochs,
        save_dir=f'/mnt/data01/AL/trained_models/{name}'
    )


def evaluate(base_model, trained_model, label_dict, original_test_dir, print_mistakes=False):

    # Tokenize data
    test = tokenize_data_for_finetuning(
        directory=f"/mnt/data01/AL/final_labelled_data/test.csv",
        hf_model=base_model,
        max_token_length=512
    )

    num_labels = len(label_dict)
    model = AutoModelForSequenceClassification.from_pretrained(trained_model, num_labels=num_labels)

    # Instantiate Trainer
    trainer = Trainer(model=model)

    predictions = trainer.predict(test)

    preds = np.argmax(predictions.predictions, axis=-1)

    if print_mistakes:
        test_df = pd.read_csv(original_test_dir)

        test_df["preds"] = preds

        fps = test_df[(test_df["label"] == 0) & (test_df["preds"] == 1)]
        fns = test_df[(test_df["label"] == 1) & (test_df["preds"] == 0)]
        tps = test_df[(test_df["label"] == 1) & (test_df["preds"] == 1)]
        tns = test_df[(test_df["label"] == 0) & (test_df["preds"] == 0)]

        print("Total mispredictions:", len(fps) + len(fns))
        print("False positives:", len(fps))
        print("False negatives:", len(fns))
        print("True positives:", len(tps))
        print("True negatives:", len(tns))

        print("\n\n")
        print("***************** FALSE POSITIVES *****************")
        for i in list(fps.index.values):
            print(fps["article"][i])
            print("*****")

        print("\n\n")
        print("***************** FALSE NEGATIVES *****************")
        for i in list(fns.index.values):
            print(fns["article"][i])
            print("*****")

    print("***Test results***")
    metric0 = load_metric("accuracy")
    print(metric0.compute(predictions=preds, references=predictions.label_ids))

    metric1 = load_metric("recall")
    print(metric1.compute(predictions=preds, references=predictions.label_ids))

    metric2 = load_metric("precision")
    print(metric2.compute(predictions=preds, references=predictions.label_ids))


if __name__ == '__main__':

    # random.seed(42)

    # data_paths = ['Labelled_data/fixed_first_1000.json',
    #               'Labelled_data/sample_11_fixed.json',
    #               'Labelled_data/sample_12_fixed.json',
    #               'Labelled_data/sample_13_fixed.json',
    #               'Labelled_data/sample_14_fixed.json',
    #               'Labelled_data/sample_15_fixed.json']

    label2int = {'Irrelevant': 0, 'On topic': 1}

    pretrained_model = 'roberta-large'

    # # Clean data
    # train_test_dev_split(
    #     ls_data_paths=data_paths,
    #     label_dict=label2int,
    #     save_dir="/mnt/data01/AL/final_labelled_data/",
    #     test_perc=0.15,
    #     eval_perc=0.15,
    #     model=pretrained_model
    # )

    # # Config hyperparameter sweep
    # sweep_configuration = {
    #     'method': 'bayes',
    #     'name': 'any_about_benefits_sweep',
    #     'metric': {'goal': 'maximize', 'name': "eval/f1"},
    #     'early_terminate': {'type': 'hyperband', 'min_iter': 100},    
    #     'parameters': 
    #         {
    #             'batch_size': {'values': [8, 16, 32, 64, 128]},
    #             'epochs': {'min': 10, 'max': 15},                                                  
    #             'lr': {'values': [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]},
    #             'max_len': {'values': [100, 256, 384, 512]}
    #         }
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='benefits_topic',  entity="stigma")

    # wandb.agent.WANDB_AGENT_MAX_INITIAL_FAILURES=20

    # wandb.agent(sweep_id, project='benefits_topic', entity="stigma", function=train_wrapper, count=200)

    evaluate(
        base_model=pretrained_model,
        trained_model='/mnt/data01/AL/trained_models/rl_8_13_1e-05_512/checkpoint-420',
        label_dict=label2int,
        original_test_dir='/mnt/data01/AL/final_labelled_data/test.csv',
        print_mistakes=True
    )

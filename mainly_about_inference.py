import os
import json
import numpy as np
import random
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_fns import find_sep_token, chunk

os.environ['TRANSFORMERS_CACHE'] = '.cache/'


def format_and_tokenize(dat, tokenization_model, max_token_length):

    corpus = []

    print("Tokenizing data ...")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    sep = find_sep_token(tokenizer)

    for art_id, art_dict in dat.items():
        corpus.append(str(art_dict['headline']) + sep + str(art_dict['article']))

    dataset = Dataset.from_dict({'corpus': corpus})

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset['corpus'], padding="max_length", truncation=True, max_length=max_token_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset



def pull_positives(tokenized_data, org_data, finetuned_topic_model, batch_size):

    # Predict
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_topic_model, num_labels=2)

    inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)

    trainer = Trainer(model=model, args=inference_args)

    preds = trainer.predict(tokenized_data)

    predictions = np.argmax(preds.predictions, axis=-1)

    # Subset to positives only
    positive_dict = {}
    count = 0
    for art_id, art in org_data.items():
        if predictions[count] == 1:
            positive_dict[art_id] = art
        count += 1

    print(f'{len(positive_dict)} articles positive out of {len(org_data)}')

    return positive_dict


if __name__ == '__main__':

    base_model='roberta-large'

    # Open data 
    # for year in range(2013, 2023):
    for year in [2013]:

        print(f"******************{year}**********************")


        with open(f'/n/home09/esilcock/mentions_benefits/mentions_benefits_{year}.json') as f: 
            mentions_benefits = json.load(f)

        # Chunk, format and tokenize
        tokenized_data = format_and_tokenize(mentions_benefits, tokenization_model=base_model, max_token_length=512)

        # Run inference
        topic_arts = pull_positives(
            tokenized_data,
            org_data=mentions_benefits,
            finetuned_topic_model='/n/home09/esilcock/NIHDAL/trained_models/mainly_about/checkpoint-280',
            batch_size=512
        )

        with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}.json', 'w') as f:
            json.dump(topic_arts, f, indent=4)

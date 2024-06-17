import os
import json
import numpy as np
import random
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_fns import find_sep_token, basic_clean, chunk

os.environ['TRANSFORMERS_CACHE'] = '.cache/'


def format_and_tokenize(dat, tokenization_model, max_token_length):

    corpus = []
    chunk_map = {}

    print("Tokenizing data ...")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    sep = find_sep_token(tokenizer)

    counter = 0
    for art_id, art_dict in dat.items():
        chunk_map[art_id] = []
        chunked_dict = chunk(art_dict, tokenizer, max_token_length)
        for ch in chunked_dict["chunks"]:
            corpus.append(str(art_dict['headline']) + sep + str(ch))
            
            chunk_map[art_id].append(counter)
            counter += 1
            
    dataset = Dataset.from_dict({'corpus': corpus})

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset['corpus'], padding="max_length", truncation=True, max_length=max_token_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset, chunk_map


def pull_positives(tokenized_data, org_data, chunk_map, finetuned_topic_model, batch_size):

    # Predict
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_topic_model, num_labels=2)

    inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)

    trainer = Trainer(model=model, args=inference_args)

    preds = trainer.predict(tokenized_data)

    predictions = np.argmax(preds.predictions, axis=-1)

    # Subset to positives only
    positive_dict = {}
    for art_id, chunk_list in chunk_map.items():
        if max([predictions[c] for c in chunk_list]) == 1: 
            positive_dict[art_id] = org_data[art_id]

    print(f'{len(positive_dict)} articles positive out of {len(org_data)}')

    return positive_dict


if __name__ == '__main__':

    base_model='roberta-large'

    # Open data 
    for year in range(2013, 2023):

        print(f"******************{year}**********************")

        with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned.json') as f:
            year_dat = json.load(f)

        year_dict = {a['ln_id']: a for a in year_dat}

        # Chunk, format and tokenize
        tokenized_data, chunk_map = format_and_tokenize(year_dat, tokenization_model=base_model, max_token_length=512)

        # Run inference
        topic_arts = pull_positives(
            tokenized_data,
            org_data=year_dict,
            chunk_map=chunk_map,
            finetuned_topic_model='/n/home09/esilcock/NIHDAL/trained_models/kw_initialisation/full_dat_16_5e-06_v2/checkpoint-840',
            batch_size=512
        )

        with open(f'/n/home09/esilcock/mentions_benefits/mentions_benefits_{year}.json', 'w') as f:
            json.dump(topic_arts, f, indent=4)

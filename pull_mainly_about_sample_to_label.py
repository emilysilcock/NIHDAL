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
    random.seed(42)

    # for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # # for num in [1, 2, 4]:

        # print(f'**{num}**')

        # # Get data
        # basic_clean(
        #     fp = f"/mnt/data01/AL/ln_data/The_Sun_(England)/The_Sun_(England)_{num}**",
        #     first_date='01-01-2013',
        #     sp=f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/"
        #     )

    # Open data 
    print("Loading data ...")
    all_articles = {}
    for num in tqdm([1, 2, 3, 4, 5, 6, 7, 8, 9]):
        with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data.json") as f:
            dat = json.load(f)
            for k, v in dat.items():
                all_articles[k] = v
        with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data_earlier.json") as f:
            dat = json.load(f)
            for k, v in dat.items():
                all_articles[k] = v

    # Take sample 
    sample_articles = {}
    for k in random.sample(all_articles.keys(), 100000):
        sample_articles[k] = all_articles[k]
    del all_articles

    # Chunk, format and tokenize
    tokenized_data, chunk_map = format_and_tokenize(sample_articles, tokenization_model=base_model, max_token_length=512)

    # Run inference
    topic_arts = pull_positives(
        tokenized_data,
        org_data=sample_articles,
        chunk_map=chunk_map,
        # finetuned_topic_model='/mnt/data01/AL/trained_models/kw_initialisation/rl_16_2e-06/checkpoint-1000',
        finetuned_topic_model='/mnt/data01/AL/trained_models/mentions_benefits',
        batch_size=512
    )

    with open(f'/mnt/data01/AL/preds/on_topic_sample_chunked_2.json', 'w') as f:
        json.dump(topic_arts, f, indent=4)

    with open(f'/mnt/data01/AL/preds/on_topic_sample_chunked_2.json') as f:
        dat = json.load(f)

    to_label = []

    for art_id, art_dict in dat.items():

        to_label.append({
            "id": art_id,
            "data": art_dict
        })

    random.shuffle(to_label)

    with open('/mnt/data01/AL/NIHDAL/data_to_label/mainly_about/mainly_about_full_final_model.json', 'w') as f:
        json.dump(to_label, f, indent=4)

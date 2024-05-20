import os
import json
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_fns import find_sep_token, basic_clean

os.environ['TRANSFORMERS_CACHE'] = '.cache/'


def format_and_tokenize(dat, tokenization_model, max_token_length):

    corpus = []

    sep = find_sep_token(tokenizer=AutoTokenizer.from_pretrained(tokenization_model))

    for art_id, art_dict in dat.items():
        corpus.append(str(art_dict['headline']) + sep + str(art_dict['article']))

    dataset = Dataset.from_dict({'corpus': corpus})

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

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

    # Subset to positives only
    predictions = np.argmax(preds.predictions, axis=-1)

    positive_dict = {}

    pred_count = 0
    for art_id, art in org_data.items():
        if predictions[pred_count] == 1:
            positive_dict[art_id] = art
        pred_count +=1

    print(f'{len(positive_dict)} articles positive out of {len(org_data)}')

    return positive_dict


if __name__ == '__main__':

    base_model='roberta-large'

    for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # for num in [1, 2, 4]:

        print(f'**{num}**')

        # Get data
        basic_clean(
            fp = f"/mnt/data01/AL/ln_data/The_Sun_(England)/The_Sun_(England)_{num}**",
            first_date='01-01-2013',
            sp=f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/"
            )

        # Format and tokenize
        with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data.json") as f:
            data = json.load(f)

        tokenized_data = format_and_tokenize(data, tokenization_model=base_model, max_token_length=512)

        # Run inference
        topic_arts = pull_positives(
            tokenized_data,
            org_data=data,
            finetuned_topic_model='/mnt/data01/AL/trained_models/rl_8_13_1e-05_512/checkpoint-420',
            batch_size=512
        )

        with open(f'/mnt/data01/AL/preds/group_{num}on_topic_earlier.json', 'w') as f:
            json.dump(topic_arts, f, indent=4)

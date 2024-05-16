import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime

from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

os.environ['TRANSFORMERS_CACHE'] = '.cache/'


def basic_clean(fp, first_date, sp):

    data_dict = {}
    not_found_dict = []

    remove_before = datetime.strptime(first_date, '%d-%m-%Y')

    paths = glob(fp)
    print(f'{len(paths)} paths to process')

    for path in tqdm(paths):

        try:
            count = int(path.split("_")[-1].split(".")[0])
        except:
            print(path)
            assert 1 == 0

        try:
            with open(path) as f:
                dat = json.load(f)

            for art in dat["value"]:

                # Parse xml
                if not art['Document']:
                    not_found_dict.append(art)
                    count += 1
                    continue

                content = art['Document']['Content']

                soup = BeautifulSoup(content, 'xml')

                # Get Date
                date = datetime.strptime(art["Date"], "%Y-%m-%dT%H:%M:%SZ").date()

                if date >= remove_before.date():   ### SWITCHED TO EARLIER DATES 
                    count += 1
                    continue

                check_date = datetime.strptime(soup.find('published').get_text(), "%Y-%m-%dT%H:%M:%SZ").date()
                assert date == check_date

                publication_date_day = soup.find('publicationDate').get('day')
                publication_date_month = soup.find('publicationDate').get('month')
                publication_date_year = soup.find('publicationDate').get('year')
                publication_date_obj = datetime.strptime(f"{publication_date_year}-{publication_date_month}-{publication_date_day}", "%Y-%m-%d")
                assert date == publication_date_obj.date()

                date = date.strftime("%Y-%m-%d")


                # Get article
                try:
                    article = soup.find('nitf:body.content').get_text(separator='\n\n')
                except:
                    article = ""

                cleaned_data = {
                    "int_id": count,
                    "ln_id": art["Document"]["DocumentId"],
                    "date": date,
                    "headline": art["Title"],
                    "article": article,
                    "newspaper": art["Source"]["Name"],
                }

                data_dict[count] = cleaned_data

                count += 1

        except:
            print(f'{path} not found')

    print(f'{len(data_dict)} articles')
    print(f'{len(not_found_dict)} articles not found')

    # Save
    os.makedirs(sp, exist_ok=True)

    with open(f"{sp}/cleaned_sample_data_earlier.json", 'w') as f:
        json.dump(data_dict, f, indent=4)

    with open(f"{sp}/not_found_sample_earlier.json", 'w') as f:
        json.dump(not_found_dict, f, indent=4)


def find_sep_token(tokenizer):

    """
    Returns sep token for given tokenizer
    """

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


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

    # for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # # for num in [1, 2, 4]:

    #     print(f'**{num}**')

    #     # Get data
    #     basic_clean(
    #         fp = f"/mnt/data01/AL/ln_data/The_Sun_(England)/The_Sun_(England)_{num}**",
    #         first_date='01-01-2013',
    #         sp=f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/"
    #         )

        # with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/cleaned_sample_data_earlier.json") as f:
        #     data = json.load(f)


    ####################

    dat = pd.read_csv('/mnt/data01/AL/not_matched_old_date_relatedbenefits.csv')

    data = {}

    for i in range(len(dat)):
        data[i] = {
            'article': dat['body.old'][i],
            'headline': dat['title.old'][i]
        }

    #######################

    # Format and tokenize
    tokenized_data = format_and_tokenize(data, tokenization_model=base_model, max_token_length=512)

    # Run inference
    topic_arts = pull_positives(
        tokenized_data,
        org_data=data,
        finetuned_topic_model='/mnt/data01/AL/trained_models/rl_8_13_1e-05_512/checkpoint-420',
        batch_size=512
    )

    print(json.dump(topic_arts, indent=4))

    # with open(f'/mnt/data01/AL/preds/group_{num}on_topic_earlier.json', 'w') as f:
    #     json.dump(topic_arts, f, indent=4)


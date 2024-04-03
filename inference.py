import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np

from bs4 import BeautifulSoup
from datetime import datetime

from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


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

            if date < remove_before:
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

    print(f'{len(data_dict)} articles')
    print(f'{len(not_found_dict)} articles not found')

    # Save
    os.makedirs(sp, exist_ok=True)

    with open(f"{sp}/cleaned_sample_data.json", 'w') as f:
        json.dump(data_dict, f, indent=4)

    with open(f"{sp}/not_found_sample.json", 'w') as f:
        json.dump(not_found_dict, f, indent=4)


def tokenize_data_for_inference(corpus, name, hf_model):

    print("**** Tokenizing data ****")

    dataset = Dataset.from_dict({name: corpus})

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset[name], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def pull_positives(org_data_pattern, tokenization_model, finetuned_topic_model, batch_size):

    #Todo: tweak to chain together multiple models (eg. editorials and topic)

    # Load data
    corpus, org_data = load_inference_data(org_data_pattern, tokenization_model)

    tokenized_data = tokenize_data_for_inference(corpus, "inf_data", tokenization_model)

    # Predict
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_topic_model, num_labels=2)

    inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)

    trainer = Trainer(model=model, args=inference_args)

    preds = trainer.predict(tokenized_data)

    # Subset to positives only
    predictions = np.argmax(preds.predictions, axis=-1)

    positive_list = []
    for i, art in enumerate(org_data):
        if predictions[i] == 1:
            positive_list.append(art)

    print(f'{len(positive_list)} articles positive out of {len(org_data)}')

    return positive_list


if __name__ == '__main__':

    for num in [111]:

        # Get data
        basic_clean(
            fp = f"/mnt/data01/AL/ln_data/The_Sun_(England)/**",
            # fp = f"/mnt/data01/AL/ln_data/The_Sun_(England)/The_Sun_(England)_{num}**",
            first_date='01-01-2013',
            sp=f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/"
            )

        # # Run inference
        # topic_arts = pull_positives(
        #     data,
        #     tokenization_model='roberta-large',
        #     finetuned_topic_model='/mnt/data01/AL/trained_models/rl_16_12_5e-05_100/checkpoint-430',
        #     batch_size=512
        # )

        # print(len(topic_arts), "articles found")

        # with open(f'/mnt/data01/AL/preds/on_topic.json', 'w') as f:
        #     json.dump(topic_arts, f, indent=4)

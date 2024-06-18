import json
from tqdm import tqdm

import numpy as np
from openai import OpenAI

# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def generate_summary(client, headline, article):

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarise this newspaper article in fewer than 25 words:"},
            {"role": "user", "content": f'{headline} /n/n {article}'},
        ]
    )
    summary = completion.choices[0].message.content

    return summary
 

def tokenize(dat, tokenization_model, max_token_length):

    print("Tokenizing data ...")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

    corpus = [art['data']['summary'] for art in dat]            
    dataset = Dataset.from_dict({'corpus': corpus})

    # Tokenize datasets
    def tokenize_function(dataset):
        return tokenizer(dataset['corpus'], padding="max_length", truncation=True, max_length=max_token_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def predict(tokenized_data, finetuned_topic_model, batch_size):

    # Predict
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_topic_model, num_labels=2)

    inference_args = TrainingArguments(output_dir="save", per_device_eval_batch_size=batch_size)

    trainer = Trainer(model=model, args=inference_args)

    preds = trainer.predict(tokenized_data)

    predictions = np.argmax(preds.predictions, axis=-1)

    return(predictions)


if __name__ == '__main__':

    # for year in range(2013, 2023):

    #     print(f"******************{year}**********************")

    #     with open(f'/n/home09/esilcock/mentions_benefits/mentions_benefits_{year}.json') as f:
    #         mentions_benefits = json.load(f)

    # with open('Labelled_data/mainly_about/final_merged_tricky_resolved.json') as f:
    #     dat = json.load(f)

    # # Summarise with GPT
    # client = OpenAI(api_key="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")

    # for art in tqdm(dat):
    #     sum = generate_summary(client, art['data']['headline'], art['data']['article'])
    #     art['data']['summary'] = sum

    # with open('Labelled_data/mainly_about/final_merged_with_summaries.json', 'w') as f:
    #     json.dump(dat, f)
    with open('Labelled_data/mainly_about/final_merged_with_summaries.json') as f:
        dat = json.load(f)

    gts = [art['annotations'][0]['result'][0]['value']['choices'][0] for art in dat]


    # Predict over summaries
    base_model='roberta-large'
    tokenized_data = tokenize(dat, tokenization_model=base_model, max_token_length=512)

    # Run inference
    preds = predict(
        tokenized_data,
        finetuned_topic_model='/n/home09/esilcock/NIHDAL/trained_models/kw_initialisation/full_dat_16_5e-06_v2/checkpoint-840',
        batch_size=512
    )

    # Eval
    tps = 0
    tns = 0
    fps = 0
    fns = 0

    for i, pred in preds:
        if pred == 1 and gts[i] == "Mainly about benefits":
            tps +=1
        elif pred == 1 and gts[i] == "Not mainly about benefits":
            fps += 1
        if pred == 0 and gts[i] == "Mainly about benefits":
            fns +=1
        elif pred == 1 and gts[i] == "Not mainly about benefits":
            tns += 1

    print(tps, tns, fps, fns)
    
    recall = tps/(tps + fns)
    precision = tps/(tps + fps)
    f1 = recall * precision * 2 /(recall + precision)
  
    print(f1, precision, recall)

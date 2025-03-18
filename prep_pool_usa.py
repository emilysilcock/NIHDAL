from tqdm import tqdm
import json

from transformers import AutoTokenizer

from data_fns import chunk

# Open all data
sample_list = []

publications = ['nyt', 'wapo', 'usa_today', 'wsj']

for publication in tqdm(publications):

    with open(f"/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/{publication}_sample.json") as f:
        sample_list.extend(json.load(f))

print("Number of articles: ", len(sample_list))


# Split into chunks

tokenization_model = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

chunked_sample = []

for s in tqdm(sample_list):
    chunked_sample.append(chunk(s, tokenizer, max_length=512, article_name="text"))

with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/usa_chunked.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4)

from tqdm import tqdm
import json
import random

from transformers import AutoTokenizer

from data_fns import chunk

# Open all data
sample_list = []

publications = ['nyt', 'wapo', 'usa_today', 'wsj']

for publication in tqdm(publications):

    with open(f"/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/{publication}_sample.json") as f:
        sample_list.extend(json.load(f))

# Add NY post 
full_dat  = []
for year in range(1997, 2022):
    with open(f"/n/netscratch/economics/Lab/esilcock/Factiva_data/Factiva_clean/NY_Post/{year}_cleaned.json") as f:
        full_dat.extend(list(json.load(f).values()))

    sample_list.extend(random.sample(full_dat, 1000))

print("Number of articles: ", len(sample_list))

with open(f"/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/Factiva_data/{publication}_sample.json") as f:
    sample_list.extend(json.load(f))




# Split into chunks

tokenization_model = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

chunked_sample = []

for s in tqdm(sample_list):
    s['article'] = s['text']
    del s['text']
    s['ln_id'] = s['goid']
    del s['goid']
    chunked_sample.append(chunk(s, tokenizer, max_length=512))


with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/usa_chunked.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4)

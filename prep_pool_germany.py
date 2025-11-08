from tqdm import tqdm
import json
import random

from transformers import AutoTokenizer

from data_fns import chunk

random.seed(42)

##  Open all data
sample_list = []

# Handelsblatt
with open(f"/n/home09/esilcock/stigma-non-take-up/0_data/raw_data/newspapers/Handelsblatt_data/Articles/handelsblatt_sample.json") as f:
    sample_list.extend(list(json.load(f).values()))


# Bild
full_dat  = []
for year in range(2011, 2024):
    with open(f"/n/netscratch/economics/Lab/esilcock/Factiva_data/Factiva_clean/Bild/{year}_cleaned.json") as f:
        full_dat.extend(list(json.load(f).values()))

sample_list.extend(random.sample(full_dat, 1000))

print("Number of articles: ", len(sample_list))


# Split into chunks

tokenization_model = 'deepset/gbert-large'
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

chunked_sample = []

for s in tqdm(sample_list):

    if 'text' in s:
        s['article'] = s['text']
        del s['text']
        s['ln_id'] = s['goid']
        del s['goid']

    chunked_sample.append(chunk(s, tokenizer, max_length=512))


with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/germany_chunked.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4, ensure_ascii=False)

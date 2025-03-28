from tqdm import tqdm
import json
import random

from transformers import AutoTokenizer

from data_fns import chunk

random.seed(42)

##  Open all data
sample_list = []

# ProQuest
publications = ['le_monde', 'les_echos']

for publication in publications:

    print(publication)

    with open(f"/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/{publication}_sample.json") as f:
        sample_list.extend(json.load(f))


# LexisNexis
publications = ["Le_Figaro", "Libération"]

for publication in publications:

    print(publication)

    full_dat = []

    for year in range(1995, 2024):
        with open(f"/n/netscratch/economics/Lab/esilcock/LexisNexis_data/LexisNexis_clean/{publication}/{year}_cleaned.json") as f:
            full_dat.extend(json.load(f))

    sample_list.extend(random.sample(full_dat, 1000))

print("Number of articles: ", len(sample_list))



# Split into chunks

tokenization_model = 'almanach/camembertav2-base'
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

chunked_sample = []

for s in tqdm(sample_list):

    if 'text' in s:
        s['article'] = s['text']
        del s['text']
        s['ln_id'] = s['goid']
        del s['goid']

    chunked_sample.append(chunk(s, tokenizer, max_length=512))


with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/france_chunked.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4)

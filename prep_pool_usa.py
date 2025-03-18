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
    s['article'] = s['text']
    del s['text']
    s['ln_id'] = s['goid']
    del s['goid']
    chunked_sample.append(chunk(s, tokenizer, max_length=512))

# Add labelled UK data 
with open('Labelled_data/kw_initialisation/full_corrected.json') as f:
    uk_data = json.load(f)

for art in uk_data:
    chunked_sample.append({
        "headline": art["data"]["headline"],
        "article": art["data"]["article"],
        "ln_id": art["data"]["ln_id"],
        "chunks": art["data"]["chunks"],
        "publisher": art["data"]["newspaper"]
    })

with open('/n/home09/esilcock/stigma-non-take-up/data/raw_data/newspapers/ProQuest_data/usa_chunked.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4)

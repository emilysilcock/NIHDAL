from tqdm import tqdm
import json

from transformers import AutoTokenizer

from data_fnsMOVED import get_pub_list, chunk

# Open all data
sample_list = []

publications = get_pub_list()

for publication in tqdm(publications):

    publication_fn = publication.replace(' ', '_')

    with open(f"Sun_data/{publication_fn}/cleaned_sample_data.json") as f:
        clean_dat = json.load(f)

    with open(f"Sun_data/sample_indices_{publication_fn}.json") as f:
        sample = json.load(f)

    # Take sample
    for s in sample:
        try:
            sample_list.append(clean_dat[str(s)])
        except:
            pass

# Split into chunks

tokenization_model = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

chunked_sample = []

for s in tqdm(sample_list):

    chunked_sample.append(chunk(s, tokenizer, max_length=512))

with open('Sun_data/chunked_sample.json', 'w') as f:
    json.dump(chunked_sample, f, indent=4)

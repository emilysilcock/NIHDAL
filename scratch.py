from tqdm import tqdm
import json
import re
import random

publications = [
    'The Sun (England)',
    'thesun.co.uk',
    'Daily Star',
    'Daily Star Online',
    'Daily Star Sunday',
    'The Daily Mail and Mail on Sunday (London)',
    'mirror.co.uk',
    'Daily Mirror',
    'The Express',
    'The Sunday Express',
    'The News of the World',
    'The Evening Standard (London)',
    'standard.co.uk',
    'The People',
    'Metro (UK)',
    'City A.M.',
    'Cityam.com',
    'The Times (London)',
    'The Sunday Times (London)',
    'thetimes.co.uk',
    'The Daily Telegraph (London)',
    'The Daily Telegraph Online',
    'The Sunday Telegraph (London)',
    'The Guardian (London)',
    'The Observer (London)',
    'i - Independent Print Ltd',
    'The Independent (United Kingdom)',
    'Liverpool Post',
    'liverpoolecho.co.uk',
    'Liverpool Echo',
]


# Open all data   
for publication in tqdm(publications):
    sample_list = []

    print(f'**{ publication}')

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

for s in sample_list:

    if s 
import json
import copy

with open('Labelled_data/kw_initialisation/sample_1.json') as f:
    dat = json.load(f)

new_labelled_data = []

for art in dat:

    chunk_id = art['data']['chunks'].index(art['data']['article'])

    art_copy = copy.deepcopy(art)

    art_copy['data']['ln_id'] = f"{art_copy['data']['ln_id']}_{chunk_id}"

    new_labelled_data.append(art_copy)

with open('Labelled_data/kw_initialisation/sample_1_with_correct_ids.json', 'w') as f:
    json.dump(new_labelled_data, f, indent=4)

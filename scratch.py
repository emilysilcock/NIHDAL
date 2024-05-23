import json
import copy

with open('Labelled_data/kw_initialisation/sample_1.json') as f:
    dat = json.load(f)

new_labelled_data = []

counter = 0
for i, art in enumerate(dat):

    if i != 0:
        if art['data']['ln_id'] == dat[i-1]['data']['ln_id']:
            counter += 1
        else:
            counter = 0

    art_copy = copy.deepcopy(art)

    art_copy['data']['ln_id'] = f"{art_copy['data']['ln_id']}_{counter}"

    new_labelled_data.append(art_copy)

with open('Labelled_data/kw_initialisation/sample_1_with_correct_ids.json', 'w') as f:
    json.dump(new_labelled_data, f, indent=4)

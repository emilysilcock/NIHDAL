import json

# Get labels from old data
old_labelled_data_paths = ['Labelled_data/fixed_first_1000.json',
                  'Labelled_data/sample_11_fixed.json',
                  'Labelled_data/sample_12_fixed.json',
                  'Labelled_data/sample_13_fixed.json',
                  'Labelled_data/sample_14_fixed.json',
                  'Labelled_data/sample_15_fixed.json']

old_labelled_data = {}
for p in old_labelled_data_paths:

    with open(p) as f:
        dat = json.load(f)

    for art in dat:
        old_labelled_data[art['data']['ln_id']] = art['annotations'][0]['result'][0]['value']['choices'][0]


# Add labels to new data 
name_bit = "first_sample"

with open(f'data_to_label/kw_initialisation/{name_bit}.json') as f:
    new_dat = json.load(f)

print(len([f for f in new_dat if f in old_labelled_data]))

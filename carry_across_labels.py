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
name_bit = "sample_17"

with open(f'data_to_label/kw_initialisation/{name_bit}.json') as f:
    new_dat = json.load(f)

all_new_ids = [f["data"]["ln_id"] for f in new_dat]

count = 0
for f in new_dat:

    if f["data"]["ln_id"].split("_")[0] in old_labelled_data:
        
        if len(f["data"]["chunks"]) == 1 or old_labelled_data[f["data"]["ln_id"].split("_")[0]] == "Irrelevant":

            count += 1

            f['annotations'] = [
                {
                    'id': count,
                    'completed_by':1,
                    'result': [
                        {
                            'value':{'choices': [old_labelled_data[f["data"]["ln_id"].split("_")[0]]]},
                            'id': 1,
                            'from_name': 'topic',
                            'to_name':'text',
                            'type': 'choices'
                        }
                    ]
                }
            ]


print(f'{count} already labelled')

with open(f'data_to_label/kw_initialisation/{name_bit}_with_old_labels.json', 'w') as f:
    json.dump(new_dat, f, indent=4)

import json

with open('Labelled_data/kw_initialisation/sample_1.json') as f:
    dat = json.load(f)

old_labelled_data = {}

counter = 0
for i, art in dat:

    if i != 0:
        if art['data']['ln_id'] == dat[i-1]['data']['ln_id']:
            counter += 1
        else:
            counter = 0 
    
    old_labelled_data[f"{art['data']['ln_id']}_{counter}"] = art['annotations'][0]['result'][0]['value']['choices'][0]        
    

# Add labels to new data 
name_bit = "first_sample_with_correct_ids"

with open(f'data_to_label/kw_initialisation/{name_bit}.json') as f:
    new_dat = json.load(f)

count = 0
for f in new_dat:
    if f["id"] in old_labelled_data:
        count += 1

        f['annotations'] = [
            {
                'id': count,
                'completed_by':1,
                'result': [
                    {
                        'value':{'choices': [old_labelled_data[f["id"]]]},
                        'id': 1,
                        'from_name': 'topic',
                        'to_name':'text',
                        'type': 'choices'
                    }
                ]
            }
        ]

print(f'{count} already labelled')

# with open(f'data_to_label/kw_initialisation/{name_bit}_with_old_labels.json', 'w') as f:
#     json.dump(new_dat, f, indent=4)

with open('Labelled_data/kw_initialisation/sample_1_with_correct_ids.json', 'w') as f:
    json.dump(new_dat, f, indent=4)

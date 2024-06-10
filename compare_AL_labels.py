import json 


# Old data
old_labelled_data_paths = ['Labelled_data/fixed_first_1000.json',
                  'Labelled_data/sample_11_fixed.json',
                  'Labelled_data/sample_12_fixed.json',
                  'Labelled_data/sample_13_fixed.json',
                  'Labelled_data/sample_14_fixed.json',
                  'Labelled_data/sample_15_fixed.json']

old_positives = {}
for p in old_labelled_data_paths:

    with open(p) as f:
        dat = json.load(f)

    for art in dat:
        if art['annotations'][0]['result'][0]['value']['choices'][0] == "On topic":
            old_positives[art['data']['ln_id']] = art['data']['article']


# New data 
new_labelled_data_paths = [
        'Labelled_data/kw_initialisation/sample_1_with_correct_ids.json',
        'Labelled_data/kw_initialisation/sample_2.json',
        'Labelled_data/kw_initialisation/sample_3.json',
        'Labelled_data/kw_initialisation/sample_4.json',
        'Labelled_data/kw_initialisation/sample_5.json',
        'Labelled_data/kw_initialisation/sample_6.json',
        'Labelled_data/kw_initialisation/sample_7.json',
        'Labelled_data/kw_initialisation/sample_8.json',
        'Labelled_data/kw_initialisation/sample_9.json',
        'Labelled_data/kw_initialisation/sample_10.json',
        'Labelled_data/kw_initialisation/sample_11.json',
        'Labelled_data/kw_initialisation/sample_12.json',
        'Labelled_data/kw_initialisation/sample_13.json',
        'Labelled_data/kw_initialisation/sample_14.json',
        'Labelled_data/kw_initialisation/sample_15.json',
        'Labelled_data/kw_initialisation/sample_16.json',
        'Labelled_data/kw_initialisation/sample_17.json',
        'Labelled_data/kw_initialisation/sample_18.json',
        'Labelled_data/kw_initialisation/sample_19.json',
        'Labelled_data/kw_initialisation/sample_20.json',
        'Labelled_data/kw_initialisation/sample_21.json',
        'Labelled_data/kw_initialisation/sample_22.json',
        'Labelled_data/kw_initialisation/sample_23.json',
        'Labelled_data/kw_initialisation/sample_24.json',
        'Labelled_data/kw_initialisation/sample_25.json',
        'Labelled_data/kw_initialisation/sample_26.json',
        'Labelled_data/kw_initialisation/sample_27.json',
        'Labelled_data/kw_initialisation/sample_28.json'
        ]

new_positives = []
new_dict = {}

for p in new_labelled_data_paths:

    with open(p) as f:
        dat = json.load(f)

    for art in dat:
        if art['annotations'][0]['result'][0]['value']['choices'][0] == "On topic":
            new_positives.append(art["data"]["ln_id"].split("_")[0])

        if art["data"]["ln_id"].split("_")[0] not in new_dict:
            new_dict[art["data"]["ln_id"].split("_")[0]] = {}
        new_dict[art["data"]["ln_id"].split("_")[0]][art["data"]["ln_id"]] = art["data"]["article"]

new_positives = list(set(new_positives))

print(len(old_positives))
count = 0
for art_id, text in old_positives.items():
    if art_id not in new_positives:
        print('missing')
        # if art_id in new_dict:
        #     count += 1
        #     if count ==5 :
        #         print('labelled some')
        #         print(text)
        #         print("***********************")
        #         print(json.dumps(new_dict[art_id], indent=2))
        # print('######################################')
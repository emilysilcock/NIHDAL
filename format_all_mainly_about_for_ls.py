import json
import random


to_label = []
for year in range(2013, 2023):

    print(f"******************{year}**********************")

    with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}.json') as f:
        dat = json.load(f)

    for art_id, art in dat.items():

        to_label.append({
            "id": art_id,
            "data": art
        })

random.shuffle(to_label)

with open(f'data_to_label/final_hand_labelling/part_1.json', 'w') as f:
    json.dump(to_label[:600], f, indent=4)
with open(f'data_to_label/final_hand_labelling/part_2.json', 'w') as f:
    json.dump(to_label[600:1300], f, indent=4)
with open(f'data_to_label/final_hand_labelling/part_3.json', 'w') as f:
    json.dump(to_label[1300:], f, indent=4)

import json
from tqdm import tqdm

editions = []
count = 0

for year in range(2013, 2023):

    print(f"******************{year}**********************")

    with open(f'/n/home09/esilcock/mentions_benefits/mentions_benefits_{year}.json') as f:
        mentions_benefits = json.load(f)

    for art_ids, art in tqdm(mentions_benefits.items()):
        if art['edition'] not in editions:
            editions.append(art['edition'])

        if art['edition'] in ['Edition 1, ', 'Edition 1']:
            print(art['date'], art['headline'])
            count += 1

print(editions)
print(count)

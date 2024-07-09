import json
from tqdm import tqdm

# date_lists = {}
# for year in range(2013, 2023):

#     print(f'*******{year}*******')

    
#     with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned.json') as f:
#         national = json.load(f)

#     with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned_non_national.json') as f:
#         non_national = json.load(f)

#     all_dat = national + non_national

#     for art in tqdm(all_dat):

#         if art['date'] not in date_lists:
#             date_lists[art['date']] = []

#         date_lists[art['date']].append({'headline': art["headline"], 'edition': art["edition"]})

# with open('all_headlines.json', 'w') as f:
#     json.dump(date_lists, f, indent=4)
with open('all_headlines.json') as f:
    date_lists = json.load(f)


print("Finding duplicates")
all_headlines = []
for date, head_dicts in tqdm(date_lists.items()):

    all_headlines.extend(list(set([a['headline'] for a in head_dicts])))

seen = set()
duplicates = {}
for item in tqdm(all_headlines):
    if item in seen:
        if item not in duplicates:
            duplicates[item] = 2
        else:
            duplicates[item] += 1
    else:
        seen.add(item)

for dup, count in duplicates.items():
    if count > 50:
        print(dup)
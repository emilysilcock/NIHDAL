import json
from tqdm import tqdm

date_lists = {}

for year in range(2013, 2023):

    print(f'*******{year}*******')

    
    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned.json') as f:
        national = json.load(f)

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned_non_national.json') as f:
        non_national = json.load(f)

    all_dat = national + non_national

    for art in tqdm(all_dat):

        if art['date'] not in date_lists:
            date_lists[art['date']] = []

        date_lists[art['date']].append({'headline': art["headline"], 'edition': art["edition"]})

with open('all_headlines.json', 'w') as f:
    json.dump(date_lists, f, indent=4)


print("Finding duplicates")
duplicates = []
for date in tqdm(date_lists):

    other_dates = [value for key, value in date_lists.items() if key != date]

    for art in date_lists[date]:

        if any(art["headline"] in head_list for head_list in other_dates):

            duplicates.append(art["headline"])

print(set(duplicates))

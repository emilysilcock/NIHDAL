import json

all_not_found = []

for year in range(2013, 2023):

    print(year)

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_not_found.json') as f:
        not_found_list = json.load(f)

    for nf_list in not_found_list:
        all_not_found.extend(nf_list)

print(len(all_not_found))

dates = []
for art_dict in all_not_found:
    dates.append(art_dict["Date"][:10])

print(dates)

with open('/n/home09/esilcock/clean_Sun_data/missing_dates.json', 'w') as f:
    json.dump(dates, f, indent=4)
import json

all_not_found = []

# for year in range(2013, 2023):
for year in range(2013, 2018):

    print(year)

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_not_found.json') as f:
        not_found_list = json.load(f)

    for nf_list in not_found_list:
        all_not_found.extend(nf_list)

print(len(all_not_found))

with open('/n/home09/esilcock/clean_Sun_data/all_not_found.json', 'w') as f:
    json.dump(all_not_found, f, indent=4)

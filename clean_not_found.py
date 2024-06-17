import json

for year in range(2013, 2023):

    print(year)

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_not_found.json') as f:
        not_found_list = json.load(f)

    print(sum(len(nf for nf in not_found_list)))

import json


# for year in range(2013, 2023):
for year in range(2014, 2023):

    print(f"******************{year}**********************")

    counter = 0

    with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}.json') as f:
        national = json.load(f)

    labelled_list = [art['article'][:50] for art in national]

    with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}_non_national.json') as f:
        non_national = json.load(f)

    for art in non_national:
        if art['article'][:50] not in labelled_list:
            counter += 1

    print(counter)
    print(len(non_national))

    

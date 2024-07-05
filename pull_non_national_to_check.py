import json


for year in range(2013, 2023):

    print(f"******************{year}**********************")


    # with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}.json') as f:
    #     national = json.load(f)

    with open(f'/n/home09/esilcock/mainly_about_benefits/mainly_about_benefits_{year}_non_national.json') as f:
        non_national = json.load(f)

    print(len(non_national))

    

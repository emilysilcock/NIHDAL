from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm 
import json


from data_fns import basic_parsing


def date_range(start_date, end_date):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    delta = end_date - start_date

    dates = []

    for i in range(delta.days + 1):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))

    return dates


edition_dict = {
    'Edition 1, Northern Ireland': 'remove',
    'Edition 1, National Edition': 'keep',
    'Edition 1, Ireland': 'remove',
    'Edition 1, Scotland': 'remove',
    'Edition 2, Northern Ireland': 'remove',
    'Edition 2, National Edition': 'keep',
    'Edition 2, Ireland': 'remove',
    'Edition 2, Scotland': 'remove',
    'Edition 3, Northern Ireland': 'remove',
    'Edition 3, National Edition': 'keep',
    'Edition 3, Ireland': 'remove',
    'Edition 3, Scotland': 'remove',
    'Edition 4, Northern Ireland': 'remove',
    'Edition 4, National Edition': 'keep',
    'Edition 4, Ireland': 'remove',
    'Edition 4, Scotland': 'remove',
    'Edition 5, Northern Ireland': 'remove',
    'Edition 5, National Edition': 'keep',
    'Edition 5, Ireland': 'remove',
    'Edition 5, Scotland': 'remove',
    'Edition 10, Scotland': 'remove',
    'Edition 3MM': 'keep',
    '': 'keep',
    'Edition 1': 'keep',
    'Edition 1GM': 'keep',
    'Edition 3GM': 'keep',
    'Edition 1, ': 'keep',
    'Edition 1GG, Super Goals': 'keep',
    'Edition 1SS, Scotland': 'remove',
    'Edition 3SS, Scotland': 'remove',
    'Edition 2UB, Ulster': 'remove'
}

for year in range(2013, 2023):
# for year in range(2021, 2023):
# for year in [2020]:

    year_list = []
    not_found_list = []

    print(f'************************{year}***********************')

    date_list = date_range(start_date=f"{year}-01-01", end_date=f"{year}-12-31")

    for date in tqdm(date_list):

        paths = glob(f'/n/holyscratch01/economics/esilcock/Sun_data/**{date}**')

        date_ids = []
        date_data = []
        for path in paths:

            with open(path) as f:
                dat = json.load(f)

            date_ids.extend([d["ResultId"] for d in dat['value']])
            date_data.extend(dat['value'])

        date_ids = list(set(date_ids))

        if not len(date_ids) == dat["@odata.count"]:
            print("*************************************************")
            print(len(date_ids))
            print(dat["@odata.count"])
            print(date)
            print("*************************************************")

        cleaned_data, not_found_data = basic_parsing(date_data)
        not_found_list.extend(not_found_data)

        national_editions = []

        for art in cleaned_data:
            try:
                # if edition_dict[art['edition']] == 'keep':     ####################
                if edition_dict[art['edition']] == 'remove':     ####################
                    year_list.append(art)
            except:
                print(list(set([a['edition'] for a in cleaned_data])))
                raise LookupError

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned_non_national.json', 'w') as f:
        json.dump(year_list, f, indent=4)
    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_not_found_non_national.json', 'w') as f:
        json.dump(not_found_list, f, indent=4)

    print('Filtered articles:', len(year_list))
    print('Not found:', len(not_found_list))

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
}

for year in range(2013, 2023):

    year_list = []
    not_found_list = []

    print(f'************************{year}***********************')

    date_list = date_range(start_date=f"{year}-01-01", end_date=f"{year}-12-31")

    for date in tqdm(date_list):

        paths = glob(f'/n/home09/esilcock/Sun_data/**{date}**')

        date_ids = []
        date_data = []
        for path in paths:

            with open(path) as f:
                dat = json.load(f)

            date_ids.extend([d["ResultId"] for d in dat['value']])
            date_data.extend(dat['value'])

        date_ids = list(set(date_ids))

        assert len(date_ids) == dat["@odata.count"]

        cleaned_data, not_found_data = basic_parsing(date_data)
        not_found_list.append(not_found_data)

        national_editions = []

        for art in cleaned_data:
            if edition_dict[art['edition']] == 'keep':
                year_list.append(art)

    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_cleaned.json', 'w') as f:
        json.dump(year_list, f, indent=4)
    with open(f'/n/home09/esilcock/clean_Sun_data/{year}_not_found.json', 'w') as f:
        json.dump(not_found_list, f, indent=4)

    print('Filtered articles:', len(year_list))
    print('Not found:', len(not_found_list))

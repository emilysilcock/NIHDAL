from datetime import datetime, timedelta
from glob import glob 
import json


def date_range(start_date, end_date):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    delta = end_date - start_date

    dates = []

    for i in range(delta.days + 1):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))

    return dates



date_list = date_range(start_date="2013-01-01", end_date="2023-12-31")

for date in date_list:

    paths = glob('')

    date_data = []
    for path in paths: 

        with open(path) as f:
            dat = json.load(f)

        date_data.extend(dat['value'])

    assert len(date_data) == dat["@odata.count"]
        
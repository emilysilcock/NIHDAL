import json
import re
from datetime import datetime
import pandas as pd

missing_dat = []
for num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    with open(f"/mnt/data01/AL/clean_data/'The_Sun_(England)'/group_{num}/not_found_sample.json") as f:
        missing_dat.extend(json.load(f))

number_words_pattern = re.compile(r'(\d+)\s*words', re.IGNORECASE)


cleaned_missing = []
for i, art in enumerate(missing_dat):


    # Get Date
    date = datetime.strptime(art["Date"], "%Y-%m-%dT%H:%M:%SZ").date()

    date = date.strftime("%Y-%m-%d")

    cleaned_data = {
        "int_id": f'e-{i}',
        "ln_id": "",
        "content_type": art["ContentType"],
        "section": art["Section"],
        "sup": "",
        "edition": "",
        "copyright": "",
        "byline": art["Byline"],
        "wordcount": int(art["WordLength"]),
        "page_number": "",
        "date": date,
        "time": "",
        "update": "",
        "headline": art["Title"],
        "lede": "",
        "article": "",
        "captions": "",
        "highlight": "",
        "newspaper": art["Source"]["Name"],
        "correction_text": "",
        "correction_date": ""
    }

    cleaned_missing.append(cleaned_data)


cleaned_df = pd.DataFrame(cleaned_missing)

cleaned_df.to_csv('missing_vals.csv')


        

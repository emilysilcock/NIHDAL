
from tqdm import tqdm
import json
import re
import random

if __name__ == '__main__':


    publications = [
        'The Sun (England)',
        'thesun.co.uk',
        'Daily Star',
        'Daily Star Online',
        'Daily Star Sunday',
        'The Daily Mail and Mail on Sunday (London)',
        'mirror.co.uk',
        'Daily Mirror',
        'The Express',
        'The Sunday Express',
        'The News of the World',
        'The Evening Standard (London)',
        'standard.co.uk',
        'The People',
        'Metro (UK)',
        'City A.M.',
        'Cityam.com',
        'The Times (London)',
        'The Sunday Times (London)',
        'thetimes.co.uk',
        'The Daily Telegraph (London)',
        'The Daily Telegraph Online',
        'The Sunday Telegraph (London)',
        'The Guardian (London)',
        'The Observer (London)',
        'i - Independent Print Ltd',
        'The Independent (United Kingdom)',
        'Liverpool Post',
        'liverpoolecho.co.uk',
        'Liverpool Echo',
    ]

    keyword_list_1 = ['social security', 'benefit fraud', 'scrounger', 'shirker', 'sponger',
                    'skiver', 'workshy', 'work-shy', 'something for nothing', 'underclass', 'benefit tourism', 'benefit tourist']

    keyword_list_2 = ['dole']

    keyword_list_3 = ['benefits', 'welfare']


    sample_size = 10

    selected_articles = []

    # Open all data   
    for publication in tqdm(publications):
        sample_list = []

        print(f'**{ publication}')

        publication_fn = publication.replace(' ', '_')

        with open(f"Sun_data/{publication_fn}/cleaned_sample_data.json") as f:
            clean_dat = json.load(f)

        with open(f"Sun_data/sample_indices_{publication_fn}.json") as f:
            sample = json.load(f)

        # Take sample
        for s in sample:
            try:
                sample_list.append(clean_dat[str(s)])
            except:
                pass

        # Look for keywords
        take_all_kw_list = []
        take_sample_kw_list = []
        for art_dict in sample_list:

            text = str(art_dict['article']).lower() + " " + str(art_dict['headline']).lower()

            if any(kw in text for kw in keyword_list_1):
                take_all_kw_list.append(art_dict)

            elif any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keyword_list_2):
                take_all_kw_list.append(art_dict)

            elif any(kw in text for kw in keyword_list_3):
                take_sample_kw_list.append(art_dict)

        selected_articles.extend(take_all_kw_list)

        remaining_sample_size = sample_size - len(take_all_kw_list)
        if len(take_sample_kw_list) > remaining_sample_size:
            selected_articles.extend(random.sample(take_sample_kw_list, sample_size - len(take_all_kw_list)))
        else:
            selected_articles.extend(take_sample_kw_list)

    # Format for label studio
    to_label = []
    for art in selected_articles:
        to_label.append({
            "id": art["ln_id"],
            "data": art
        })

    random.shuffle(to_label)

    with open('data_to_label/kw_initialisation/first_sample.json', 'w') as f:
        json.dump(to_label, f, indent=4)

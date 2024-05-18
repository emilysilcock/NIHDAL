
from tqdm import tqdm
import json

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

    # Open all data

    keyword_list = ['benefits', 'welfare', 'social security', 'dole', 'benefit fraud', 'scrounger', 'shirker', 'sponger', 
                    'skiver', 'workshy', 'work-shy', 'something for nothing', 'underclass', 'benefit tourism', 'benefit tourist']

    scrounger_list = []

    for publication in tqdm(publications):

        publication_fn = publication.replace(' ', '_')

        with open(f"Sun_data/{publication_fn}/cleaned_sample_data.json") as f:
            clean_dat = json.load(f)

        for art_id, art_dict in clean_dat.items():
            if any(kw in str(art_dict['article']).lower() for kw in keyword_list) or any(kw in str(art_dict['headline']).lower() for kw in keyword_list):
                scrounger_list.append(art_dict)

    with open('scrounger_list.json', 'w') as f:
        json.dump(scrounger_list, f, indent=4)

    print(len(scrounger_list))

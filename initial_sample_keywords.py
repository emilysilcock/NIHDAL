
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

    # keyword_list = ['benefits', 'welfare', 'social security', 'dole', 'benefit fraud', 'scrounger', 'shirker', 'sponger',
    #                 'skiver', 'workshy', 'work-shy', 'something for nothing', 'underclass', 'benefit tourism', 'benefit tourist']
    keyword_list = ['dole']

    # Open all data
    sample_list = []

    for publication in tqdm(publications):

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

    for kw in keyword_list:
        kw_list = []
        for art_dict in sample_list:
            if kw in str(art_dict['article']).lower()  or kw in str(art_dict['headline']).lower() :
            # if any(kw in str(art_dict['article']).lower() for kw in keyword_list) or any(kw in str(art_dict['headline']).lower() for kw in keyword_list):
                kw_list.append(art_dict)

        print(f'{kw}: {len(kw_list)}')

    with open('scrounger_list.json', 'w') as f:
        json.dump(kw_list, f, indent=4)

    print(len(kw_list))

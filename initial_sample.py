import json
from tqdm import tqdm
from glob import glob

from bs4 import BeautifulSoup
import re
from datetime import datetime
import random

import numpy as np
from transformers import pipeline, AutoTokenizer

## Clean data

def clean_data(publications):

    with open('The_Sun_(England)/The_Sun_(England)_1401.json') as f:
        dat = json.load(f)

    # content = dat["value"][0]['Document']['Content']
    # soup = BeautifulSoup(content, 'xml')

    # # print(soup.prettify())

    # [tag.name for tag in soup.find_all(True)]

    known_tags = [
        'identifier',
        'positionSection',
        'articleDocHead',
        'positionSequence',
        'articleDoc',
        'wordCount',
        'bodyText',
        'published',
        'publicationInfo',
        'title',
        'inlineObject',
        'classCode',
        'locator',
        'metadata',
        'name',
        'logo',
        'publicationDate',
        'body',
        'p',
        'locatorKey',
        'hl1',
        'id',
        'classificationGroup',
        'itemInfo',
        'className',
        'updated',
        'source',
        'copyright',
        'locatorParameters',
        'author',
        'edition',
        'publicationLogo',
        'dateText',
        'body.end',
        'entry',
        'person',
        'hedline',
        'keyValue',
        'date',
        'sourceSectionInfo',
        'content',
        'keyName',
        'classification',
        'classificationItem',
        'parameter',
        'byline',
        'publicationName',
        'body.head',
        'body.content',
        'nameText',
        'graphic',
        'hl2',
        'media',
        'caption',
        'email',
        'url',
        'correction',
        'correctionDescription',
        'correctionDate',
        'correctionStatusText',
        'highlight',
        'br',
        'timeReceived',
        'table',
        'tgroup',
        'colspec',
        'tbody',
        'row',
        'list',
        'listItem',
        'label',
        'pagination',
        'pageSequence',
        'page',
        'pageSequence',
        'pagination',
        'h',
        'sup',
        'byttl',
        'emphasis',
        'thead',
        'positionSubsection',
        'contactInfo',
        'note',
        'abstract',
        'fixture'
    ]

    # Data cleaning

    number_words_pattern = re.compile(r'(\d+)\s*words', re.IGNORECASE)

    for publication in publications:

        data_dict = {}
        not_found_dict = []

        print(publication)

        paths = glob(f"{publication.replace(' ', '_')}/{publication.replace(' ', '_')}**")
        print(len(paths))

        paths = files_saved_before_date(paths, datetime(2024, 2, 3))
        print(len(paths))

        for path in tqdm(paths):

            try:
                count = int(path.split("_")[-1].split(".")[0])
            except:
                print(path)
                assert 1 == 0

            with open(path) as f:
                dat = json.load(f)

            for art in dat["value"]:

                # Parse xml
                if not art['Document']:
                    not_found_dict.append(art)
                    count += 1
                    continue

                content = art['Document']['Content']

                soup = BeautifulSoup(content, 'xml')

                tags = [tag.name for tag in soup.find_all(True)]

                for tag in tags:
                    if tag not in known_tags:
                        print(f'*********************{tag}**********************')
                        print(soup.prettify())
                        raise ValueError("Tag unknown")

                # Get article
                try:
                    article = soup.find('nitf:body.content').get_text(separator='\n\n')
                except:
                    article = None

                lede_paragraph = soup.find('p', {'nitf:lede': 'true'})
                if lede_paragraph:
                    lede = lede_paragraph.get_text(separator='\n\n')
                else:
                    lede = None

                caption_text = soup.find('p', {'caption': 'true'})
                if caption_text:
                    captions = caption_text.get_text(separator='\n\n')
                else:
                    captions = None

                highlight_text = soup.find('highlight')
                if highlight_text:
                    highlight = highlight_text.get_text(separator='\n\n')
                else:
                    highlight = None


                sup_text = soup.find('sup')
                if sup_text:
                    sup = sup_text.get_text(separator='\n\n')
                else:
                    sup = None


                # Corrections
                correction = soup.find('correctionDescription')

                if correction:
                    correction_text = correction.get_text(separator='\n')

                    if soup.find('correctionDate'):

                        correction_date_day = soup.find('correctionDate').get('day')
                        correction_date_month = soup.find('correctionDate').get('month')
                        correction_date_year = soup.find('correctionDate').get('year')

                        try:
                            correction_date = datetime.strptime(f"{correction_date_year}-{correction_date_month}-{correction_date_day}", "%Y-%m-%d").strftime("%Y-%m-%d")
                        except:
                            correction_date = None

                    else:
                        correction_date = None

                elif soup.find('contactInfo'):

                    correction_text = soup.find('contactInfo').get_text(separator='\n')
                    correction_date = None

                else:
                    correction_text = None
                    correction_date = None




                # Get edition
                edition = "".join([tag.get_text(separator=' ') for tag in soup.find_all('edition')])

                # Get copyright
                copyright = "".join([tag.get_text(separator=' ') for tag in soup.find_all('copyright')])

                # Get number of words
                overview = art['Overview']
                match_number_words = number_words_pattern.search(overview)
                if match_number_words:
                    number_words = match_number_words.group(1)

                    try:
                        assert int(soup.find('wordCount')['number']) == int(art["WordLength"])
                    except:
                        print(json.dumps(art, indent=2))
                    assert int(number_words) == art["WordLength"]

                else:
                    number_words = None


                # Page number
                page_number = " ".join([tag.get_text(separator=' ') for tag in soup.find_all('positionSequence')])

                # Get Date
                date = datetime.strptime(art["Date"], "%Y-%m-%dT%H:%M:%SZ").date()

                check_date = datetime.strptime(soup.find('published').get_text(), "%Y-%m-%dT%H:%M:%SZ").date()
                assert date == check_date

                publication_date_day = soup.find('publicationDate').get('day')
                publication_date_month = soup.find('publicationDate').get('month')
                publication_date_year = soup.find('publicationDate').get('year')
                publication_date_obj = datetime.strptime(f"{publication_date_year}-{publication_date_month}-{publication_date_day}", "%Y-%m-%d")
                assert date == publication_date_obj.date()

                date = date.strftime("%Y-%m-%d")


                update = datetime.strptime(soup.find('dc:date', {'dateType': 'last-updated'}).get_text(), "%Y-%m-%d").date().strftime("%Y-%m-%d")

                # Time
                time_text = soup.find('timeReceived')
                if time_text:
                    time = time_text.get_text(separator='\n\n')
                else:
                    time = None


                cleaned_data = {
                    "int_id": count,
                    "ln_id": art["Document"]["DocumentId"],
                    "content_type": art["ContentType"],
                    "section": art["Section"],
                    "sup": sup,
                    "edition": edition,
                    "copyright": copyright,
                    "byline": art["Byline"],
                    "wordcount": int(art["WordLength"]),
                    "page_number": page_number,
                    "date": date,
                    "time": time,
                    "update": update,
                    "headline": art["Title"],
                    "lede": lede,
                    "article": article,
                    "captions": captions,
                    "highlight": highlight,
                    "newspaper": art["Source"]["Name"],
                    "correction_text": correction_text,
                    "correction_date": correction_date
                }

                data_dict[count] = cleaned_data

                count += 1

        print(f'{len(not_found_dict)} articles not found')

        with open(f"{publication.replace(' ', '_')}/cleaned_sample_data.json", 'w') as f:
            json.dump(data_dict, f, indent=4)

        with open(f"{publication.replace(' ', '_')}/not_found_sample.json", 'w') as f:
            json.dump(not_found_dict, f, indent=4)


def find_sep_token(model_name):

    """
    Returns sep token for given tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


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

    # clean_dat = clean_data(publications)

    # Load model
    classifier = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0)

    chosen_positives = []
    chosen_negatives = []

    sample_list = []

    # Open all data
    for publication in tqdm(publications):

        publication_fn = publication.replace(' ', '_')

        with open(f"Sun_data/{publication_fn}/cleaned_sample_data.json") as f:
            clean_dat = json.load(f)

        with open(f"Sun_data/sample_indices_{publication_fn}.json") as f:
            sample = json.load(f)

        # Take sample
        sample_dict = {}
        found_sample = []
        for s in sample:
            try:
                sample_dict[str(s)] = clean_dat[str(s)]
                found_sample.append(str(s))
            except:
                pass

        print(f'{len(sample_dict)} articles in sample')

        # Format texts
        sep = find_sep_token('facebook/bart-large')
        texts = [str(article['headline']) + sep + str(article['article']) for article in sample_dict.values()]

        # Run inference
        candidate_labels = ["welfare benefits", "a different topic"]
        hypothesis_template = "This example is about {}."
        batch_size = 64

        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            sequences = texts[i:i + batch_size]
            result = classifier(sequences, candidate_labels, hypothesis_template=hypothesis_template)
            results.extend(result)

        print(f"{len([r for r in results if r['labels'][0] == candidate_labels[0]])} positives")
        print(f"{len([r for r in results if r['labels'][0] == candidate_labels[1]])} negatives")

        positives = []
        negatives = []        
        for i, res in enumerate(results):

            if res['labels'][0] == candidate_labels[0]:
                positives.append(sample_dict[found_sample[i]])

            else:
                negatives.append(sample_dict[found_sample[i]])


        keyword_list = ['benefits', 'welfare', 'social security', 'dole', 'benefit fraud', 'scrounger', 'shirker', 'sponger',
                'skiver', 'workshy', 'work-shy', 'something for nothing', 'underclass', 'benefit tourism', 'benefit tourist']

        ######
        non_scrounger_list = []
        for art_dict in positives:
            if any(kw in str(art_dict['article']).lower() for kw in keyword_list) or any(kw in str(art_dict['headline']).lower() for kw in keyword_list):
                continue
            else:
                non_scrounger_list.append(art_dict)

        print(f'{len(non_scrounger_list)} found by ZSC but not by keywords')

        with open(f'temp/{publication_fn}_zsc_extra.json', 'w') as f:
            json.dump(non_scrounger_list, f, indent=4)

        ######


    #     # Sample from results
    #     if len(positives) > 2:
    #         chosen_positives.extend(random.sample(positives, 2))
    #     else:
    #         chosen_positives.extend(positives)
    #     chosen_negatives.extend(random.sample(negatives, 2))

    # with open('positives.json', 'w') as f:
    #     json.dump(chosen_positives, f, indent=4)
    # with open('negatives.json', 'w') as f:
    #     json.dump(chosen_negatives, f, indent=4)

    # with open('positives.json') as f:
    #     chosen_positives = json.load(f)
    # with open('negatives.json') as f:
    #     chosen_negatives = json.load(f)

    # to_label = []
    # for i, art in enumerate(chosen_positives + chosen_negatives):
    #     to_label.append({
    #         "id": i,
    #         "data": art
    #     })

    # random.shuffle(to_label)

    # with open('first_sample.json', 'w') as f:
    #     json.dump(to_label, f, indent=4)

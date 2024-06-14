from tqdm import tqdm
from glob import glob
from statistics import variance
import math

import json

from bs4 import BeautifulSoup
from datetime import datetime


def get_pub_list():

    return [
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


def find_sep_token(tokenizer):

    """
    Returns sep token for given tokenizer
    """

    if 'eos_token' in tokenizer.special_tokens_map:
        sep = " " + tokenizer.special_tokens_map['eos_token'] + " " + tokenizer.special_tokens_map['sep_token'] + " "
    else:
        sep = " " + tokenizer.special_tokens_map['sep_token'] + " "

    return sep


# def basic_clean(fp, sp):

#     data_dict = {}
#     not_found_dict = []

#     paths = glob(fp)
#     print(f'{len(paths)} paths to process')

#     for path in tqdm(paths):

#         try:
#             count = int(path.split("_")[-1].split(".")[0])
#         except:
#             print(path)
#             assert 1 == 0

#         try:
#             with open(path) as f:
#                 dat = json.load(f)

#             for art in dat["value"]:

#                 # Parse xml
#                 if not art['Document']:
#                     not_found_dict.append(art)
#                     count += 1
#                     continue

#                 content = art['Document']['Content']

#                 soup = BeautifulSoup(content, 'xml')

#                 # Get Date
#                 date = datetime.strptime(art["Date"], "%Y-%m-%dT%H:%M:%SZ").date()

#                 if date >= remove_before.date():   ### SWITCHED TO EARLIER DATES 
#                     count += 1
#                     continue

#                 check_date = datetime.strptime(soup.find('published').get_text(), "%Y-%m-%dT%H:%M:%SZ").date()
#                 assert date == check_date

#                 publication_date_day = soup.find('publicationDate').get('day')
#                 publication_date_month = soup.find('publicationDate').get('month')
#                 publication_date_year = soup.find('publicationDate').get('year')
#                 publication_date_obj = datetime.strptime(f"{publication_date_year}-{publication_date_month}-{publication_date_day}", "%Y-%m-%d")
#                 assert date == publication_date_obj.date()

#                 date = date.strftime("%Y-%m-%d")


#                 # Get article
#                 try:
#                     article = soup.find('nitf:body.content').get_text(separator='\n\n')
#                 except:
#                     article = ""

#                 cleaned_data = {
#                     "int_id": count,
#                     "ln_id": art["Document"]["DocumentId"],
#                     "date": date,
#                     "headline": art["Title"],
#                     "article": article,
#                     "newspaper": art["Source"]["Name"],
#                 }

#                 data_dict[count] = cleaned_data

#                 count += 1

#         except:
#             print(f'{path} not found')

#     print(f'{len(data_dict)} articles')
#     print(f'{len(not_found_dict)} articles not found')

#     if sp:
#         # # Save
#         # os.makedirs(sp, exist_ok=True)

#         # with open(f"{sp}/cleaned_sample_data_earlier.json", 'w') as f:
#         #     json.dump(data_dict, f, indent=4)

#         # with open(f"{sp}/not_found_sample_earlier.json", 'w') as f:
#         #     json.dump(not_found_dict, f, indent=4)

#     return data_dict, not_found_dict 



def basic_parsing(list_of_articles):

    not_found_list = []
    cleaned_articles = []
    for art in list_of_articles:

        # Parse xml
        if not art['Document']:
            not_found_list.append(art)
            continue

        content = art['Document']['Content']

        soup = BeautifulSoup(content, 'xml')

        # Get edition
        edition = "".join([tag.get_text(separator=' ') for tag in soup.find_all('edition')])

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

        # Get article
        try:
            article = soup.find('nitf:body.content').get_text(separator='\n\n')
        except:
            article = ""

        cleaned_data = {
            "ln_id": art["Document"]["DocumentId"],
            "date": date,
            "edition": edition, 
            "headline": art["Title"],
            "article": article,
            # "newspaper": art["Source"]["Name"],
        }

        cleaned_articles.append(cleaned_data)

    return cleaned_articles, not_found_list


def chunk(art_dict, tokenizer, max_length=512):

    headline_length = len(tokenizer.tokenize(str(art_dict["headline"])))

    art_length = len(tokenizer.tokenize(str(art_dict["article"])))

    # If short enough to be one chunk
    if headline_length + art_length + 3 < max_length:

        art_dict['chunks'] = [art_dict["article"]]

    # Otherwise partition
    else:

        chunk_max_length = max_length - headline_length - 3

        paragraphs = art_dict["article"].split("\n\n")
        para_lengths = [len(tokenizer.tokenize(para)) + 2  for para in paragraphs]

        # Deal with long paragraphs - mostly TV schedules and lists
        for i, para in enumerate(paragraphs):
            if para_lengths[i] > chunk_max_length:

                p_num_chunks = math.ceil(para_lengths[i]/chunk_max_length)
                p_chunk_length = para_lengths[i]//p_num_chunks

                p_tokens = tokenizer.tokenize(para)

                p_chunks = [p_tokens[i * p_chunk_length:(i + 1) * p_chunk_length] for i in range(p_num_chunks)]

                if para_lengths[i] % p_num_chunks != 0:
                    p_chunks[-1].extend(p_tokens[p_num_chunks * p_chunk_length:])

                p_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in p_chunks]

                paragraphs[i:i+1] = p_texts
                
                p_lengths = [p_chunk_length] * p_num_chunks
                if para_lengths[i] % p_num_chunks != 0:
                    p_lengths[-1] += para_lengths[i] - (p_num_chunks * p_chunk_length)

                para_lengths[i:i+1] = p_lengths

        # Chunk
        all_chunks = []
        para_dict = {i: para_lengths[i] for i in range(len(para_lengths))}
        while len(para_dict) > 0:
            running_sum = 0
            ch = []
            for i, para_len in para_dict.items():
                if running_sum + para_len > chunk_max_length:
                    break
                ch.append(i)
                running_sum += para_len

            all_chunks.append(ch)

            # Stop if reached end
            if ch[-1] == len(para_lengths) - 1:
                for j in ch:
                    del para_dict[j]

            # Create overlap
            elif para_lengths[ch[-1]] + para_lengths[ch[-1] + 1] > chunk_max_length:
                for j in ch:
                    del para_dict[j]
            elif (para_lengths[ch[-1]] < 10) and (para_lengths[ch[-2]] + para_lengths[ch[-1]] + para_lengths[ch[-1] + 1] <= chunk_max_length):
                for j in ch[:-2]:
                    del para_dict[j]
            else:
                for j in ch[:-1]:
                    del para_dict[j]

        art_dict['chunks'] = ["\n\n".join([paragraphs[i] for i in ch]) for ch in all_chunks]

    return art_dict

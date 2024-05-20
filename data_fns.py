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


def basic_clean(fp, first_date, sp):

    data_dict = {}
    not_found_dict = []

    remove_before = datetime.strptime(first_date, '%d-%m-%Y')

    paths = glob(fp)
    print(f'{len(paths)} paths to process')

    for path in tqdm(paths):

        try:
            count = int(path.split("_")[-1].split(".")[0])
        except:
            print(path)
            assert 1 == 0

        try:
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

                # Get Date
                date = datetime.strptime(art["Date"], "%Y-%m-%dT%H:%M:%SZ").date()

                if date >= remove_before.date():   ### SWITCHED TO EARLIER DATES 
                    count += 1
                    continue

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
                    "int_id": count,
                    "ln_id": art["Document"]["DocumentId"],
                    "date": date,
                    "headline": art["Title"],
                    "article": article,
                    "newspaper": art["Source"]["Name"],
                }

                data_dict[count] = cleaned_data

                count += 1

        except:
            print(f'{path} not found')

    print(f'{len(data_dict)} articles')
    print(f'{len(not_found_dict)} articles not found')

    # Save
    os.makedirs(sp, exist_ok=True)

    with open(f"{sp}/cleaned_sample_data_earlier.json", 'w') as f:
        json.dump(data_dict, f, indent=4)

    with open(f"{sp}/not_found_sample_earlier.json", 'w') as f:
        json.dump(not_found_dict, f, indent=4)


def chunk(art_dict, tokenizer, max_length=512):

    headline_length = len(tokenizer.tokenize(art_dict["headline"]))

    art_length = len(tokenizer.tokenize(art_dict["article"]))

    # If short enough to be one chunk
    if headline_length + art_length + 3 < max_length:

        art_dict['chunks'] = [art_dict["article"]]

    # Otherwise partition
    else:

        chunk_max_length = max_length - headline_length - 3
        num_chunks = math.ceil(art_length/chunk_max_length)

        paragraphs = art_dict["article"].split("\n\n")
        para_lengths = [len(tokenizer.tokenize(para)) + 2  for para in paragraphs]
        print(para_lengths)
        print(len(para_lengths))
        print(json.dumps(paragraphs, indent=2))

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

        possible_partitions = partition_list(para_lengths, n_sublists=num_chunks, max_len=chunk_max_length)

        if len(possible_partitions) == 0:
            num_chunks += 1
            possible_partitions = partition_list(para_lengths, n_sublists=num_chunks, max_len=chunk_max_length)

        best_partition = find_partition_with_lowest_variance(possible_partitions)

        # Add overlaps
        overlapped_partition = expand_overlaps(best_partition, para_lengths, chunk_max_length)

        art_dict['chunks'] = ["\n\n".join([paragraphs[i] for i in part]) for part in overlapped_partition]


    return art_dict
                

def partition_list(input_list, n_sublists, max_len=512):
    def is_valid_partition(partition):
        return all(sum(sublist) <= max_len for sublist in partition)
    
    def get_partitions(lst, n):
        if n == 1:
            yield [lst]
        else:
            for i in range(1, len(lst)):
                for p in get_partitions(lst[i:], n - 1):
                    yield [lst[:i]] + p

    valid_partitions = []
    for partition in get_partitions(input_list, n_sublists):
        if is_valid_partition(partition):
            valid_partitions.append(partition)
    
    return valid_partitions


def calculate_variance_of_partition(partition):
    sums = [sum(sublist) for sublist in partition]

    return variance(sums)


def find_partition_with_lowest_variance(partitions):
    min_variance = float('inf')
    best_partition = None
    for partition in partitions:
        current_variance = calculate_variance_of_partition(partition)
        if current_variance < min_variance:
            min_variance = current_variance
            best_partition = partition

    # Return indices
    current_index = 0
    result = []

    for sublist in best_partition:
        new_sublist = []
        for _ in sublist:
            new_sublist.append(current_index)
            current_index += 1
        result.append(new_sublist)

    return result 


def expand_overlaps(partition, length_list, max_len):

    # Expand first partition
    first_partition = partition[0]

    current_sum = sum([length_list[i] for i in first_partition])
    next_para = first_partition[-1] + 1

    while current_sum + length_list[next_para] <= max_len:

        first_partition.append(next_para)
        current_sum = sum([length_list[i] for i in first_partition])
        next_para = first_partition[-1] + 1

    # Expand last partition 
    last_partition = partition[-1]

    current_sum = sum([length_list[i] for i in last_partition])
    next_para = last_partition[0] -1

    while current_sum + length_list[next_para] <= max_len:

        last_partition.insert(0, next_para)
        current_sum = sum([length_list[i] for i in last_partition])
        next_para = last_partition[0] -1
    
    # Expand middle partitions
    middle_partitions = partition[1:-1]

    for middle_partition in middle_partitions:


        current_sum = sum([length_list[i] for i in middle_partition])
        next_para_forward = middle_partition[-1] + 1
        next_para_backward = middle_partition[0] - 1

        while current_sum + length_list[next_para_forward] <= max_len or current_sum + length_list[next_para_backward] <= max_len:

            if current_sum + length_list[next_para_forward] <= max_len:
                middle_partition.append(next_para_forward)
                current_sum = sum([length_list[i] for i in middle_partition])
                next_para_forward = middle_partition[-1] + 1
            
            if current_sum + length_list[next_para_backward] <= max_len:
                middle_partition.insert(0, next_para_backward)
                current_sum = sum([length_list[i] for i in middle_partition])
                next_para_backward = middle_partition[0] - 1
            
    return partition

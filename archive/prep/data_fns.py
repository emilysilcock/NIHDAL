import math


def chunk(art_dict, tokenizer, max_length=512, article_name = "article"):

    headline_length = len(tokenizer.tokenize(str(art_dict["headline"])))

    art_length = len(tokenizer.tokenize(str(art_dict[article_name])))

    # If short enough to be one chunk
    if headline_length + art_length + 3 < max_length:

        art_dict['chunks'] = [art_dict[article_name]]

    # Otherwise partition
    else:

        chunk_max_length = max_length - headline_length - 3

        paragraphs = art_dict[article_name].split("\n\n")
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


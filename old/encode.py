import numpy as np


def encode(token_texts, word2idx, max_len):

    input_ids = []
    for text in token_texts:
        #PADDING
        text += ['<pad>'] *(max_len-len(text))

        #REPLACE TOKENS WITH IDS
        input_id = [word2idx.get(token) for token in text]
        input_ids.append(input_id)

    return np.array(input_ids)
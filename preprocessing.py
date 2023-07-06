from nltk import word_tokenize
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import os
import pickle



def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.
    
    Args:
        texts (List[str]): List of text data
    
    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    idx = 2
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        tokenized_texts.append(tokenized_sent)

        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len



def encode(tokenized_texts, word2idx, max_len):
 

    input_ids = []
    for tokenized_sent in tokenized_texts:
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)
    
    return np.array(input_ids)


def load_pretrained_vectors(word2idx, fname):
    
    if os.path.exists('embeddings/annomi.pkl'):
        with open('embeddings/annomi.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    
    else:
        print("Loading pretrained vectors...")
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())

        embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
        embeddings[word2idx['<pad>']] = np.zeros((d,))

        count = 0
        for line in tqdm_notebook(fin):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in word2idx:
                count += 1
                embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

        print(f"There are {count} / {len(word2idx)} pretrained vectors found.")
        
        with open('embeddings/annomi.pkl', 'wb') as fp:
            pickle.dump(embeddings, fp)


        return embeddings
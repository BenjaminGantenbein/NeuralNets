import pandas as pd
import spacy
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import Word2Vec
import fasttext
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import pickle
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

from tqdm import tqdm_notebook


nlp = spacy.load('en_core_web_sm')



##PREPROCESSING FUNCTIONS

def tokenize(x):
    return [w.lemma_.lower() for w in nlp(x) if not w.is_punct and not w.is_stop]

def get_word2vec(token, model):
    if token in model.wv.index_to_key:
        return model.wv.get_vector(token)
    else:
        # handle out-of-vocabulary tokens
        return [0] * model.vector_size


### MAIN IMPORT METHOD
def process(path):
    
    #IMPORTING DATA
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ["sent", "label"] 
    
    #PREPROCESSING
    data['tokens'] = data['sent'].apply(lambda x: tokenize(x))
    data['preprocessed'] = data['tokens'].apply(lambda x : ' '.join(x))
    corpus = [x for sublist in data['preprocessed'] for x in sublist]
    
    #VECTORIZING
    vectorizer = TfidfVectorizer(min_df=0.01, # at min 1% of docs
                        max_df=.9,  
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1,3))
    X = vectorizer.fit_transform(data['preprocessed'])
    y = data['label'].values
    
    X = X.toarray()
    return X, y

#MAIN IMPORT METHOD USING GENSIM

def process_word2vec(path):
    
    if os.path.isfile('model.pkl'):
       with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)
    else:
        dataset = api.load("text8")
        model = Word2Vec(dataset)
        with open('model.pkl', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #IMPORTING DATA

    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ["sent", "label"] 
    
    #PREPROCESSING
    data['tokens'] = data['sent'].apply(lambda x: tokenize(x))
    
    #VECTORIZING
    embeddingsSize=128
    model=Word2Vec(data['tokens'], vector_size=embeddingsSize, window=5, min_count=1, workers=4)

    data['preprocessed'] = data['tokens'].apply(lambda x : [get_word2vec(token, model) for token in x])
    data = data[data['preprocessed'].apply(lambda x : len(x) != 0)]  
  
    torch_tensors = [torch.tensor(seq) for seq in data['preprocessed'].values]
    
    padded_sequences = pad_sequence(torch_tensors, batch_first=True, padding_value=0)
    
    X = padded_sequences
    
    
    X = X.view(-1, 18, embeddingsSize)

    y = data['label'].values
    
    
    return X, y


def process_FastText(path):
    #READING
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ["sent", "label"] 

    max_len = 0
    tokenized_texts = []
    
    word2idx = {}
    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sent in data['sent']:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, np.array(data['label']), word2idx, max_len
    
def loadEmbeddings(word2idx, file):
    
    #READ PRETRAINED VECTORS, n is  NR of WORD, d is VECTORSIZE
    fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    #INITALIZE EMBEDDINGS AT RANDOM
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d, ))

    #LOAD PRERTAINED WEIGHTS IN TO WORD VECTORS
    count = 0
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count +=1 
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)
    print(f"Found {count} / of {len(word2idx)} pretrained TOKENS !!")

    return embeddings
        







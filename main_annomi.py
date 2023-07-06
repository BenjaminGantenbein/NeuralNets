from preprocessing import (tokenize, encode, load_pretrained_vectors)
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from dataloader import data_loader
from torch import nn
from initializer import initialize_annomi_model
from train import train
from model_evaluater import evaluate_model
import numpy as np
import random
import nltk

random_seed = 200

def drop_word_length(df, drop_pct, length):

    indices_with_word_length_2 = df.index[(df['sentence_lengths'] == length) & (df['main_therapist_behaviour'] == 0)].tolist()
    
    drop_count = int(len(indices_with_word_length_2) * drop_pct)
    
    indices_to_drop = np.random.choice(indices_with_word_length_2, drop_count, replace=False).tolist()
    
    df_filtered = df.loc[~df.index.isin(indices_to_drop)].reset_index(drop=True)
    
    return df_filtered

def equalize_category_counts(df, col_name):

    category_counts = df[col_name].value_counts()

    min_count = category_counts.min()

    if random_seed is not None:
        random.seed(random_seed)


    indices_to_keep = []
    for category, count in category_counts.iteritems():
        indices = df.index[df[col_name] == category].tolist()
        if count > min_count:
            indices_to_keep.extend(np.random.choice(indices, min_count, replace=False).tolist())
        else:
            indices_to_keep.extend(indices)

    df_filtered = df.loc[indices_to_keep].reset_index(drop=True)

    return df_filtered


data = pd.read_csv('/Users/benjamingantenbein/Library/CloudStorage/Dropbox/ETH Informatik/6 Semester/Data Science in Techno/NeuralNets/data/AnnoMI/dataset.csv')
therapist_data = data[data['interlocutor']=='therapist']
client_data = data[data['interlocutor'] =='client']
label_map = {"other":0, "question":1, "reflection":2, "therapist_input":3}

def replace_label(x):
	return label_map[x]

therapist_data['sentence_lengths'] =   therapist_data['utterance_text'].apply(lambda x: len(nltk.word_tokenize(x)))

therapist_data['main_therapist_behaviour'] = therapist_data['main_therapist_behaviour'].apply(replace_label).astype(int)
therapist_data = drop_word_length(therapist_data,.999, 2)
therapist_data = drop_word_length(therapist_data,.99, 1)
therapist_data = drop_word_length(therapist_data,.99, 0)

therapist_data = equalize_category_counts(therapist_data, 'main_therapist_behaviour')


data_sent = list(therapist_data['utterance_text'])
labels = list(therapist_data['main_therapist_behaviour'])




print("Tokenizing...\n")
tokenized_texts, word2idx, max_len = tokenize(data_sent)
input_ids = encode(tokenized_texts, word2idx, max_len)

print(max_len)

embeddings = load_pretrained_vectors(word2idx, 'wiki-news-300d-1M.vec')
embeddings = torch.tensor(embeddings)

input_ids, test_inputs, labels, test_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)



train_dataloader, val_dataloader = data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=30)

filter_sizes = [2, 3, 4]
num_filters = [20, 50, 50]

loss_fn = nn.CrossEntropyLoss()

model, optimizer = initialize_annomi_model(embeddings, filter_sizes=filter_sizes, num_filters=num_filters, modelType='lstm_model', vocab_size=len(word2idx),
                                    embed_dim=300,
                                    learning_rate=0.001,
                                    hidden_size=128,
				                    num_classes=4, 
                                    dropout =0.1)


print("ENTERING TRAING")
train(model, optimizer, train_dataloader, val_dataloader, epochs=20, loss_fn=loss_fn)
print("FINISHED TRAINING")
model.load_state_dict(torch.load('best_model.pth'))

print("EVALUATING")
evaluate_model(model, test_inputs, test_labels, loss_fn)
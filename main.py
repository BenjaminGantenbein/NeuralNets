from preprocessing import (tokenize, encode, load_pretrained_vectors)
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from dataloader import data_loader
from torch import nn
from initializer import initialize_model
from train import train
from model_evaluater import evaluate_model

data = pd.read_csv('data/sentiment_labeled_sentences/imdb_labelled.txt', sep='\t', header=None)
data.columns = ["sent", "label"] 
data_sent = list(data['sent'])
labels = list(data['label'])



print("Tokenizing...\n")
tokenized_texts, word2idx, max_len = tokenize(data_sent)
input_ids = encode(tokenized_texts, word2idx, max_len)

embeddings = load_pretrained_vectors(word2idx, 'wiki-news-300d-1M.vec')
embeddings = torch.tensor(embeddings)

input_ids, test_inputs, labels, test_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)



train_dataloader, val_dataloader = data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=50)

filter_sizes = [2, 3, 4]
num_filters = [20, 20, 20]

loss_fn = nn.CrossEntropyLoss()
"""
model, optimizer = initilize_model(embeddings, vocab_size=len(word2idx),
                                      embed_dim=300,
                                      learning_rate=0.25,
                                      dropout=0.5)
"""
#model, optimizer = initilize_model(embeddings, modelType='ffn_model', embed_dim=300, learning_rate=0.5, hidden_size=64, dropout=0.5)

"""
model, optimizer = initilize_model(embeddings, modelType='lstm_model',hidden_size=128,
                                      embed_dim=300,
                                      learning_rate=0.001,
                                      dropout=0.5)

"""

model, optimizer = initialize_model(embeddings, modelType='ffn_model',
                                      embed_dim=300,
                                      hidden_size=128,
                                      learning_rate=.9)

print("ENTERING TRAING")
train(model, optimizer, train_dataloader, val_dataloader, epochs=40, loss_fn=loss_fn)
print("FINISHED TRAINING")
model.load_state_dict(torch.load('best_model.pth'))

print("EVALUATING")
evaluate_model(model, test_inputs, test_labels, loss_fn)
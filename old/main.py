from old.import_preprocessing import process, process_word2vec, process_FastText, loadEmbeddings
from old.eval import evaluate
from old.trainer import train, train_two
from old.encode import encode
from old.initializer import initilize_model

from architectures.torch.feed_forward import Feed_Forward
from architectures.torch.feed_forward_word2vec import Feed_Forward_Emb
from architectures.torch.cnn import CNN_Text
from architectures.torch.lstm import LSTM

import pandas as pd
import torch
from torch import nn
from old.dataloader import load_data, GenericDataset, data_loader


if torch.cuda.is_available():
    device ='cuda'
else:
    device ='cpu'



token_texts, y, word2idx, max_length = process_FastText('data/sentiment_labeled_sentences/yelp_labelled.txt')
encoded_texts = encode(token_texts, word2idx, max_length)
embeddings = loadEmbeddings(word2idx, 'wiki-news-300d-1M.vec')
embeddings = torch.tensor(embeddings)

train_loader, val_loader = data_loader(encoded_texts, y , batch_size=50)

#model = Feed_Forward(X.shape[1])

model, optimizer = initilize_model(embeddings, vocab_size=len(word2idx),
                                      embed_dim=300,
                                      learning_rate=0.25,
                                      dropout=0.5)

loss_fn = nn.CrossEntropyLoss()
#train(model=model, epochs=100, optimizer=optimizer, loss_fn=loss_fn, train_loader = train_loader, valid_loader=val_loader)
train_two(model=model, optimizer =optimizer, loss_fn= loss_fn, train_dataloader =train_loader, val_dataloader=val_loader, epochs=10)



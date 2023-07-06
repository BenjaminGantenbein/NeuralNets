from torch.utils.data import (Dataset, DataLoader, TensorDataset, RandomSampler)
from sklearn.model_selection import train_test_split
import math
import numpy as np
import torch

#DEFINING DATASET CLASS
class GenericDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def load_data(X, y):

    ###
    tsize = math.ceil(0.1 * len(y))
    X_train, X_valid, y_train, y_valid = train_test_split(X, np.array(y), test_size=tsize)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=tsize)
    train_dataset = GenericDataset(X_train, y_train)
    valid_dataset = GenericDataset(X_valid, y_valid)
    test_dataset = GenericDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)



    return train_loader, valid_loader, test_loader


def data_loader(input_ids, labels,  batch_size=50):

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42)
    
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, val_inputs, train_labels, val_labels])

    #TRAINING DATALOADER
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size=batch_size)

    #VALIDATION DATALOADER
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = RandomSampler(train_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)




    return train_dataloader, val_dataloader
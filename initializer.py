import torch.optim as optim
from architectures.torch.cnn import CNN_Text
from architectures.torch.feed_forward import FFN_Text, FFN_Text_binary
from architectures.torch.lstm import LSTM_Text, LSTM_Text2, LSTM_Text3, LSTM_Text4
from architectures.torch.transformer import TransformerClassifier

device = 'cpu'


def initialize_model(embeddings, modelType,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    hidden_size=128,
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    if modelType == 'cnn_model':
        model = CNN_Text(pretrained_embedding=embeddings,
                            freeze_embedding=freeze_embedding,
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            filter_sizes=filter_sizes,
                            num_filters=num_filters,
                            num_classes=num_classes,
                            dropout=dropout)
        optimizer = optim.Adadelta(model.parameters(),
                               lr=learning_rate,
                               rho=0.95)
        
    elif modelType == 'ffn_model':
        model = FFN_Text_binary(pretrained_embedding =embeddings,
                            freeze_embedding=False,
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            hidden_size=hidden_size,
                            num_classes=num_classes,
                            dropout=dropout 
                            )
        optimizer = optim.Adadelta(model.parameters(), lr = learning_rate)
    elif modelType == 'lstm_model':
        model = LSTM_Text4(pretrained_embedding =embeddings,
                            freeze_embedding=False,
                            hidden_size=hidden_size,
                            num_classes=num_classes
                            )
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif modelType == 'transformer_model':
        model = TransformerClassifier(pretrained_embedding=embeddings,
                                      num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Send model to `device` (GPU/CPU)
    else:
        NameError("MODEL WRONGLY ASSIGNED")
    model.to(device)

    # Instantiate Adadelta optimizer
    

    return model, optimizer

def initialize_annomi_model(embeddings, modelType,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    hidden_size=512,
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    if modelType == 'cnn_model':
        model = CNN_Text(pretrained_embedding=embeddings,
                            freeze_embedding=freeze_embedding,
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            filter_sizes=filter_sizes,
                            num_filters=num_filters,
                            num_classes=num_classes,
                            dropout=dropout)
       
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        """
        
        optimizer = optim.Adadelta(model.parameters(),
                               lr=learning_rate,
                               rho=0.95)
        """
    elif modelType == 'ffn_model':
        model = FFN_Text(pretrained_embedding =embeddings,
                            freeze_embedding=False,
                            vocab_size=vocab_size,
                            embed_dim=embed_dim,
                            hidden_size=hidden_size,
                            num_classes=num_classes,
                            dropout=dropout
                            )
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    elif modelType == 'lstm_model':
        model = LSTM_Text4(pretrained_embedding =embeddings,
                            freeze_embedding=False,
                            num_classes=num_classes
                            )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    elif modelType == 'transformer_model':
        model = TransformerClassifier(pretrained_embedding=embeddings, 
                                      num_classes=num_classes, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Send model to `device` (GPU/CPU)
    model.to(device)

    # Instantiate Adadelta optimizer
    

    return model, optimizer
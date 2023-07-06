import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_Text(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 hidden_size=512,
                 num_layers=1,
                 num_classes=2,
                 dropout=0.5):
   

        super(LSTM_Text, self).__init__()
        # Embedding layer
        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
  

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)


        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):

        x_embed = self.embedding(input_ids).float()

        _, (h_n, c_n) = self.lstm(x_embed)
        x = h_n.squeeze()

        logits = self.fc(x)

        return logits
    
class LSTM_Text2(nn.Module):

    def __init__(self, pretrained_embedding=None, freeze_embedding=False, dimension=128, num_classes=2, dropout=0.5):
        super(LSTM_Text2, self).__init__()

        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(dimension, num_classes)

    def forward(self, input_ids):

        x_embed = self.embedding(input_ids).float()
        text_len = torch.sum(input_ids != 1, dim=1)

        packed_input = pack_padded_sequence(x_embed, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)

        return text_fea



class LSTM_Text3(nn.Module):
    
    def __init__(self, pretrained_embedding=None, freeze_embedding=False, hidden_size=128, num_classes=4):
        super(LSTM_Text3, self).__init__()
        self.vocab_size, self.embed_dim = pretrained_embedding.shape

        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_size,
                            num_layers=1
                            )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
       
        input_ids = input_ids.transpose(0, 1)
        embedded = self.embedding(input_ids).float()
        
        output, (hidden, cell) = self.lstm(embedded)
       
        hidden.squeeze_(0)
        output = self.fc(hidden)
       
        return output


class LSTM_Text4(nn.Module):
    
    def __init__(self, pretrained_embedding=None, freeze_embedding=False, hidden_size=128, num_classes=4, dropout=0.2):
        super(LSTM_Text4, self).__init__()
        self.vocab_size, self.embed_dim = pretrained_embedding.shape

        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional = True
                            )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).float()
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        hidden = self.relu(hidden)
        logits = self.fc(hidden)
        output = F.softmax(logits, dim=1)
        return output


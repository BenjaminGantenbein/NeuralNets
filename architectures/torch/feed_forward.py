import torch
import torch.nn.functional as F
from torch import nn


class FFN_Text(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 hidden_size=128,
                 num_classes=2,
                 dropout=0.5):


        super(FFN_Text, self).__init__()
  
        self.num_classes = num_classes
        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        self.fc1 = nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
      
        x_embed = self.embedding(input_ids).float()

        x = F.relu(self.fc1(x_embed.mean(axis=1))).float()

        x = F.relu(self.fc2(self.dropout(x)))

        x = F.relu(self.fc3(self.dropout(x)))
        logits = self.fc4(self.dropout(x))

        return logits
    
class FFN_Text_binary(nn.Module):
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 hidden_size=128,
                 num_classes=2,
                 dropout=0.5):


        super(FFN_Text_binary, self).__init__()
  
        self.num_classes = num_classes
        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        self.fc1 = nn.Linear(self.embed_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
      
        x_embed = self.embedding(input_ids).float()

        x = F.relu(self.fc1(x_embed.mean(axis=1))).float()

        x = F.relu(self.fc3(self.dropout(x)))
        logits = self.fc4(self.dropout(x))

        return logits


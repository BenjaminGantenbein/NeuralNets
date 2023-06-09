from torch import nn 
from transformer import BertModel

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, model):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(model)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)
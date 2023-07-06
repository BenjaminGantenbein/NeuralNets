import torch.nn as nn
import torch.nn.functional as F
import torch

class TransformerClassifier(nn.Module):
    def __init__(self, pretrained_embedding, num_classes=2, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding_dim = pretrained_embedding.shape[1]
        self.transformer_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=5,
                dim_feedforward=128,
                dropout=self.dropout
            ),
            num_layers=2
        )
        self.fc = nn.Linear(self.embedding_dim, num_classes)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=True,
            padding_idx=0
        )

    def forward(self, input_ids):
        # Get embeddings from `input_ids`. Output shape: (batch_size, max_sent_length, embedding_dim)
        embeddings = self.embedding(input_ids).float()
        # Permute embeddings to match input shape requirement of `nn.TransformerEncoder`. Output shape: (max_sent_length, batch_size, embedding_dim)
        embeddings = embeddings.permute(1, 0, 2)
        # Apply transformer. Output shape: (max_sent_length, batch_size, embedding_dim)
        transformer_output = self.transformer_model(embeddings)
        # Mean pooling. Output shape: (batch_size, embedding_dim)
        pooled_output = torch.mean(transformer_output, dim=0)
        # Compute logits. Output shape: (batch_size, num_classes)
        logits = self.fc(pooled_output)
        return logits

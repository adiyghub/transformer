import torch
import torch.nn as nn

class SequenceEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, padding_idx, max_len, dropout_p, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.dropout_p = dropout_p
        self.device = device
        self.token_embedding = nn.Embedding(
            input_dim,
            hidden_dim,
            padding_idx
        )
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs_len = inputs.shape[1]
        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1)
        pos = pos.to(self.device)
        self.scale = self.scale.to(self.device)
        embedded = (self.token_embedding(inputs) * self.scale) + self.position_embedding(pos)
        output = self.dropout(embedded)
        
        return output
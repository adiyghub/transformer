import torch
import torch.nn as nn
from multi_attention import MultiAttention

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim=512, qkv_dim=64, n_heads=8, dropout_p=0.1):
        super(EncoderBlock, self).__init__()
        self.dropout_p = dropout_p
        self.multi_attention = MultiAttention(embedding_dim, qkv_dim, n_heads)
        # Define the feed-forward network as nn.Sequential 
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # First linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(embedding_dim * 4, embedding_dim) # Second linear layer
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, source_embedding, source_mask):
        # Perform multi-head self-attention on the source embedding
        attention_output, _ = self.multi_attention(self_embedding=source_embedding, mask=source_mask)
        attention_output = self.dropout1(attention_output)

        # Apply layer normalization and residual connection
        residual1 = self.layer_norm1(source_embedding + attention_output)

        # Pass the output through the feed-forward network
        feed_forward_output = self.feed_forward(residual1)
        feed_forward_output = self.dropout2(feed_forward_output)

        # Apply layer normalization and residual connection again
        output = self.layer_norm2(residual1 + feed_forward_output)

        return output

class Encoder(nn.Module):
    def __init__(self, sequence_to_embedding, input_dim, embedding_dim=512, qkv_dim=64, n_heads=8, dropout_p=0.1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.sequence_to_embedding = sequence_to_embedding
        self.encoder_block = EncoderBlock(embedding_dim, qkv_dim, n_heads, dropout_p)

    def forward(self, source_input, source_mask):
        embedding = self.sequence_to_embedding(source_input)
        output = self.encoder_block(embedding, source_mask)
        return output
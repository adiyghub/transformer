import torch
import torch.nn as nn
from multi_attention import MultiAttention

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim=512, qkv_dim=64, n_heads=8, dropout_p=0.1):
        super(DecoderBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.decoder_attention = MultiAttention(embedding_dim, qkv_dim, n_heads) # self attention layer
        self.encoder_decoder_attention = MultiAttention(embedding_dim, qkv_dim, n_heads) # cross attention layer
        # Define the feed-forward network as nn.Sequential 
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # First linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(embedding_dim * 4, embedding_dim) # Second linear layer
        )
        self.decoder_attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_decoder_attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, target_embedding, encoder_output, target_mask, source_mask):
        # Perform multi-head self-attention on the source embedding
        self_attention_output, _ = self.decoder_attention(self_embedding=target_embedding, mask=target_mask)
        self_attention_output = self.dropout(self_attention_output)
        self_attention_output = self.decoder_attention_layer_norm(target_embedding + self_attention_output)

        # Perform multi-head cross-attention using target embedding and encoder output
        cross_attention_output, _ = self.encoder_decoder_attention(self_embedding=self_attention_output, cross_embedding=encoder_output, mask=source_mask)
        cross_attention_output = self.dropout(cross_attention_output)
        cross_attention_output = self.encoder_decoder_attention_layer_norm(target_embedding + cross_attention_output)

        # Pass the output through the feed-forward network
        feed_forward_output = self.feed_forward(cross_attention_output)
        feed_forward_output = self.dropout(feed_forward_output)

        # Apply layer normalization and residual connection again
        output = self.feed_forward_layer_norm(cross_attention_output + feed_forward_output)

        return output

class Decoder(nn.Module):
    def __init__(self, sequence_to_embedding, output_dim, embedding_dim=512, qkv_dim=64, n_heads=8, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.sequence_to_embedding = sequence_to_embedding
        self.decoder_block = DecoderBlock(embedding_dim, qkv_dim, n_heads, dropout_p)
        self.output_linear = nn.Linear(embedding_dim, output_dim, bias=False)
        self.output_linear.weight = nn.Parameter(self.sequence_to_embedding.token_embedding.weight)

    def forward(self, target_input, encoder_output, target_mask, source_mask):
        target_embedding = self.sequence_to_embedding(target_input)
        output = self.decoder_block(target_embedding, encoder_output, target_mask, source_mask)
        output = self.output_linear(output)
        
        return output
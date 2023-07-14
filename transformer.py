import torch 
import torch.nn as nn
from build_dataset import BuildDataset
from vocabulary import TokenVocabulary
from sequence_embedding import SequenceEmbedding
from encoder import Encoder
from decoder import Decoder
from utils import construct_mask

class Transformer(nn.Module):
    def __init__(self, source_dim, 
                       target_dim,
                       max_len,
                       source_pad_idx,
                       target_pad_idx,
                       embedding_dim=512, 
                       n_heads=8, 
                       dropout_p=0.1):
        super(Transformer, self).__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.max_len = max_len
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.n_heads = n_heads
        self.qkv_dim = embedding_dim//n_heads

        self.sequence_to_embedding = SequenceEmbedding(self.source_dim, embedding_dim, self.source_pad_idx, self.max_len, dropout_p)

        self.encoder = Encoder(self.sequence_to_embedding, self.source_dim, embedding_dim, self.qkv_dim, n_heads, dropout_p)
        self.decoder = Decoder(self.sequence_to_embedding, self.target_dim, embedding_dim, self.qkv_dim, n_heads, dropout_p)
    
    def forward(self, source_sequence_batch, target_sequence_batch):

        source_mask, target_mask = construct_mask(source_sequence_batch, target_sequence_batch, self.source_pad_idx, self.target_pad_idx)
        
        # Perform multi-head self-attention on the source
        source_encoder_output = self.encoder(source_sequence_batch, source_mask)

        # Perform multi-head cross-attention using target and encoder output
        output = self.decoder(target_sequence_batch, source_encoder_output, target_mask, source_mask)

        return output
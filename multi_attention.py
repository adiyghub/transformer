import torch
import torch.nn as nn
import math

def attention_mech(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, mask_matrix=None):
    batch_size = query.shape[0]
    query_sequence_length = query.shape[1]
    original_dim = query.shape[2]

    # Storing them separately to reshape kv in cross attention, in self attention we can use the same sequence lenght for qkv 
    key_sequence_length = key.shape[1]
    value_sequence_length = value.shape[1]

    q_dim , k_dim, v_dim = query.shape[-1]//n_heads, key.shape[-1]//n_heads, value.shape[-1]//n_heads

    query = query.reshape(batch_size, n_heads, query_sequence_length, q_dim)
    key = key.reshape(batch_size, n_heads, key_sequence_length, k_dim)
    value = value.reshape(batch_size, n_heads, value_sequence_length, v_dim)
    scores = torch.matmul(query, key.transpose(2, 3))/math.sqrt(k_dim)

    if mask_matrix is not None:
        scores = scores.masked_fill(mask_matrix == 0, -1e9)

    softmax_scores = torch.softmax(scores, dim=-1)

    value_weighted_scores = torch.matmul(softmax_scores, value)

    # the reshaping works for both self attention and cross attention 
    value_weighted_scores = value_weighted_scores.reshape(batch_size, query_sequence_length, original_dim)

    return value_weighted_scores, scores


class MultiAttention(nn.Module):
    def __init__(self, embedding_dim, qkv_dim, n_heads=8):
        super(MultiAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        
        self.q_Linear = nn.Linear(embedding_dim, qkv_dim * n_heads)
        self.k_Linear = nn.Linear(embedding_dim, qkv_dim * n_heads)
        self.v_Linear = nn.Linear(embedding_dim, qkv_dim * n_heads)
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, self_embedding, cross_embedding=None, mask=None):
        if cross_embedding is None:
            q_output, k_output, v_output = self.q_Linear(self_embedding), self.k_Linear(self_embedding), self.v_Linear(self_embedding)
            output, attention_matrix = attention_mech(q_output, k_output, v_output, self.n_heads, mask)
            output = self.output_linear(output)
        else:
            q_output, k_output, v_output = self.q_Linear(self_embedding), self.k_Linear(cross_embedding), self.v_Linear(cross_embedding)
            output, attention_matrix = attention_mech(q_output, k_output, v_output, self.n_heads, mask)
            output = self.output_linear(output)
            
        return output, attention_matrix
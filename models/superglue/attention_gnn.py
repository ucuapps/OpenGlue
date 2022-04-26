import torch
import torch.nn as nn

from models.superglue import get_attention_mechanism
from ..utils import FeedForwardNet


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention='softmax'):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim // num_heads

        self.attention_func = get_attention_mechanism(embed_dim, attention)

        self.num_heads = num_heads
        self.in_proj_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.in_proj_k = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.in_proj_v = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        self.out_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        query = self.in_proj_q(query).view(batch_size, self.num_heads, self.embed_dim, -1)
        key = self.in_proj_k(key).view(batch_size, self.num_heads, self.embed_dim, -1)
        value = self.in_proj_v(value).view(batch_size, self.num_heads, self.embed_dim, -1)

        out, attention = self.attention_func(query, key, value)

        # apply out projection
        out = out.view(batch_size, self.num_heads * self.embed_dim, -1)
        return self.out_proj(out), attention


class ResidualAttentionMessagePropagation(nn.Module):
    def __init__(self, embed_dim, num_heads, attention='softmax', use_offset=False):
        super(ResidualAttentionMessagePropagation, self).__init__()
        self.use_offset = use_offset

        self.mha = MultiheadAttention(embed_dim, num_heads, attention)
        self.fc = FeedForwardNet(2 * embed_dim, 2 * embed_dim, embed_dim)

    def forward(self, desc_q, desc_kv):
        """
        desc_q - descriptors used to make query
        desc_kv- descriptors used to make key and value, might be same as desc_q
        if SELF attention or from different image if CROSS attention
        """
        message, _ = self.mha(desc_q, desc_kv, desc_kv)
        # add offset attention https://arxiv.org/abs/2012.09688
        if self.use_offset:
            message = torch.cat([desc_q - message, message], dim=1)
        else:
            message = torch.cat([desc_q, message], dim=1)
        return desc_q + self.fc(message)


class DescriptorsSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention='softmax', use_offset=False):
        super(DescriptorsSelfAttention, self).__init__()
        self.module = ResidualAttentionMessagePropagation(embed_dim, num_heads, attention, use_offset)

    def forward(self, desc0, desc1):
        desc0 = self.module(desc0, desc0)
        desc1 = self.module(desc1, desc1)
        return desc0, desc1


class DescriptorsCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention='softmax', use_offset=False):
        super(DescriptorsCrossAttention, self).__init__()
        self.module = ResidualAttentionMessagePropagation(embed_dim, num_heads, attention, use_offset)

    def forward(self, desc0, desc1):
        desc0 = self.module(desc0, desc1)
        desc1 = self.module(desc1, desc0)
        return desc0, desc1


class GraphAttentionNet(nn.Module):
    def __init__(self, num_stages, embed_dim, num_heads, attention='softmax', use_offset=False):
        super(GraphAttentionNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_stages):
            self.layers.extend((
                DescriptorsSelfAttention(embed_dim, num_heads, attention, use_offset),
                DescriptorsCrossAttention(embed_dim, num_heads, attention, use_offset),
            ))

    def forward(self, desc0, desc1):
        for l in self.layers:
            desc0, desc1 = l(desc0, desc1)
        return desc0, desc1

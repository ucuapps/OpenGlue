import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_attention(query, key, value):
    embed_dim = query.size(2)
    # get attention scores
    query = query.transpose(2, 3).contiguous()  # B,H,D,N -> B,H,N,D
    attention = torch.matmul(query, key) * embed_dim ** -0.5  # B,H,N,D @ B,H,D,M = B,H,N,M
    attention = attention.softmax(dim=-1)

    # multiply attention by values
    value = value.transpose(2, 3).contiguous()  # B,H,D,M -> B,H,M,D
    out = torch.matmul(attention, value).transpose(2, 3).contiguous()

    return out, attention


def linear_attention_elu(query, key, value):
    eps = 1e-6
    query = F.elu(query) + 1 + eps
    key = F.elu(key) + 1 + eps
    return linear_attention(query, key, value)


def linear_attention(query, key, value):
    value = value.transpose(2, 3).contiguous()  # B,H,D,M -> B,H,M,D
    kv = torch.matmul(key, value)  # B,H,D,M @ B,H,M,D = B,H,D,D
    key_norm = key.sum(3, keepdim=True)  # B,H,D,M -> B,H,D,1

    query = query.transpose(2, 3).contiguous()  # B,H,D,N -> B,H,N,D
    out = torch.matmul(query, kv)  # B,H,N,D @ B,H,D,D = B,H,N,D
    out_norm = torch.matmul(query, key_norm)  # B,H,N,D @ B,H,D,1 = B,H,N,1
    out = out / out_norm
    out = out.transpose(2, 3).contiguous()  # B,H,N,D -> B,H,D,N

    return out, None


class FavorAttention(nn.Module):
    def __init__(self, embed_dim, num_orthogonal_features=None, eps=1e-6):
        super(FavorAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_orthogonal_features = num_orthogonal_features if num_orthogonal_features is not None \
            else int(math.log(embed_dim) * embed_dim)
        self.eps = eps

        projection_matrix = self.sample_orthogonal_random_vectors(self.num_orthogonal_features, self.embed_dim)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, query, key, value):
        query = self.randomized_kernel(query, is_query=True)
        key = self.randomized_kernel(key, is_query=False)

        return linear_attention(query, key, value)

    @staticmethod
    def sample_orthogonal_random_vectors(num_rows, num_cols, device=None):
        num_blocks = math.ceil(num_rows / num_cols)
        unstructured_block = torch.randn(num_blocks, num_cols, num_cols, device=device)
        norm = unstructured_block.norm(dim=-1).view(-1, 1)
        q, r = torch.linalg.qr(unstructured_block, mode='reduced')

        # select num_rows vectors
        q = q.transpose(-1, -2).view(-1, num_cols)
        q = q[:num_rows, :]
        norm = norm[:num_rows, :]

        # multiply unit vectors by norm of random normal vectors
        # norm = torch.distributions.Chi2(q.new_tensor(num_cols)).sample((num_rows, 1)).sqrt()

        return q * norm

    @torch.no_grad()
    def resample_projection(self):
        projection_matrix = self.sample_orthogonal_random_vectors(self.num_orthogonal_features, self.embed_dim)
        self.projection_matrix.copy_(projection_matrix)

    def randomized_kernel(self, x, is_query=True):
        raise NotImplementedError


class GeneralizedFavorAttention(FavorAttention):
    def __init__(self, embed_dim, kernel_func, num_orthogonal_features=None, eps=1e-4):
        super(GeneralizedFavorAttention, self).__init__(embed_dim, num_orthogonal_features,
                                                        eps)
        self.kernel_func = kernel_func

    def randomized_kernel(self, x, is_query=True):
        embed_dim = x.size(2)
        x = x * embed_dim ** -0.25
        x = torch.matmul(self.projection_matrix, x)  # K,D @ B,H,D,N -> B,H,K,N
        return self.kernel_func(x) + self.eps


class SoftmaxFavorAttention(FavorAttention):
    def randomized_kernel(self, x, is_query=True):
        data_normalizer = (x.shape[-2] ** -0.25)
        ratio = (self.projection_matrix.shape[0] ** -0.5)

        data_dash = torch.matmul(self.projection_matrix, data_normalizer * x)  # K,D @ B,H,D,N -> B,H,K,N

        diag_data = x ** 2  # B,H,D,N
        diag_data = torch.sum(diag_data, dim=-2, keepdim=True)  # B,H,1,N
        diag_data = (diag_data / 2.0) * (data_normalizer ** 2)  # B,H,1,N

        if is_query:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data -
                              torch.amax(data_dash, dim=-2, keepdim=True)) + self.eps)
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + self.eps)

        return data_dash


if __name__ == '__main__':
    device = 'cuda:0'

    x = torch.randn(1000, 1000).to(device)

    for i in range(20):
        y = x @ x

    query = torch.randn(10, 4, 64, 1024).to(device)
    key = torch.randn(10, 4, 64, 1025).to(device)
    value = torch.randn(10, 4, 64, 1025).to(device)

    favor_attention = GeneralizedFavorAttention(embed_dim=64, kernel_func=nn.ReLU(inplace=True),
                                                num_orthogonal_features=128).to(device)

    y1, _ = softmax_attention(query, key, value)
    y2, _ = linear_attention_elu(query, key, value)
    y3, _ = favor_attention(query, key, value)
    print(y1.shape, y2.shape, y3.shape)

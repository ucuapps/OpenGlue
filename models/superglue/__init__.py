import torch.nn as nn


from models.utils import FeedForwardNet, FeedForwardNetSiren
from .attention import softmax_attention, linear_attention_elu, GeneralizedFavorAttention, SoftmaxFavorAttention

methods = {
    "FeedForwardNet": FeedForwardNet,
    "FeedForwardNetSiren": FeedForwardNetSiren
}


def get_attention_mechanism(embed_dim, attention_name):
    """Returns attention function based on the name of attention type"""
    if attention_name == 'softmax':
        return softmax_attention
    elif attention_name == 'linear':
        return linear_attention_elu
    elif attention_name == 'favor_relu':
        return GeneralizedFavorAttention(
            embed_dim,
            kernel_func=nn.ReLU(inplace=True),
            num_orthogonal_features=2 * embed_dim,
            eps=1e-8
        )
    elif attention == 'favor_softmax':
        return SoftmaxFavorAttention(
            embed_dim,
            num_orthogonal_features=2 * embed_dim,
            eps=1e-8
        )
    else:
        ValueError(f'Attention type {attention_name} is not supported.')


def get_positional_encoder(model_name):
    """
    Create method form configuration
    """
    if model_name not in methods:
        raise NameError('{} module was not found among positional encoders. Please choose one of the following '
                        'methods: {}'.format(model_name, ', '.join(methods.keys())))

    return methods[model_name]


__all__ = ['get_positional_encoder', 'get_attention_mechanism']

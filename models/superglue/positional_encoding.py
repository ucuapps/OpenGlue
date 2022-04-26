import torch
import torch.nn as nn

from models.superglue import get_positional_encoder


class MLPPositionalEncoding(nn.Module):
    def __init__(self, output_size, side_info_size=1, encoder_name='FeedForwardNet', hidden_layers_sizes=None):
        super(MLPPositionalEncoding, self).__init__()
        if hidden_layers_sizes is None:
            hidden_layers_sizes = []

        input_size = side_info_size + 2  # add 2 dimensions for xy coordinates
        self.encoder = get_positional_encoder(encoder_name)(input_size, *hidden_layers_sizes, output_size)

    def forward(self, kpts, side_info):
        b, n, _ = kpts.size()
        input = torch.cat([kpts, side_info], dim=-1).transpose(1, 2).contiguous()
        return self.encoder(input)

import math

import torch
import torch.nn as nn

from .attention_gnn import GraphAttentionNet
from .optimal_transport import log_otp_solver
from .positional_encoding import MLPPositionalEncoding


class SuperGlue(nn.Module):
    def __init__(self, config):
        super(SuperGlue, self).__init__()
        self.config: dict = config

        self.positional_encoding = MLPPositionalEncoding(**self.config['positional_encoding'])
        self.attention_gnn = GraphAttentionNet(**self.config['attention_gnn'])
        self.residual = self.config.get('residual', False)
        self.no_descriptors = self.config.get('no_descriptors', False)
        if self.residual:
            self.mix_coefs = nn.parameter.Parameter(torch.zeros(self.config['descriptor_dim'], 1))
        self.linear_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1)
        self.dustbin_score = nn.Parameter(torch.tensor(self.config['dustbin_score_init']))

        weights_path = self.config.get('weights', None)
        if weights_path is not None:
            print('SuperGlue loading... ', self.load_state_dict(torch.load(str(weights_path), map_location='cpu')))

    def forward(self, data):
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        ldesc0, ldesc1 = data['local_descriptors0'], data['local_descriptors1']
        side_info0, side_info1 = data['side_info0'], data['side_info1']
        ldesc0, ldesc1 = ldesc0.transpose(1, 2).contiguous(), ldesc1.transpose(1, 2).contiguous()

        if 'image0' in data and 'image1' in data:
            image0_size, image1_size = data['image0'].size(), data['image1'].size()
        else:
            image0_size, image1_size = data['image0_size'][::-1], data['image1_size'][::-1]

        # 1. Normalize keypoints
        kpts0, kpts1 = self.normalize_keypoints(kpts0, image0_size), self.normalize_keypoints(kpts1, image1_size)

        # 2. Add positional encoding based on xy-coordinates and side info
        pe0, pe1 = self.positional_encoding(kpts0, side_info0), self.positional_encoding(kpts1, side_info1)
        if self.no_descriptors:
            gdesc0, gdesc1 = self.attention_gnn(
                desc0=pe0,
                desc1=pe1
            )
        else:
            # 3. Transform local descriptors to global
            gdesc0, gdesc1 = self.attention_gnn(
                desc0=ldesc0 + pe0,
                desc1=ldesc1 + pe1
            )

        # 4. Linear transformation of global descriptors
        gdesc0, gdesc1 = self.linear_proj(gdesc0), self.linear_proj(gdesc1)
        if self.residual:
            alpha = torch.sigmoid(self.mix_coefs)
            gdesc0 = alpha * gdesc0 + (1.0 - alpha) * ldesc0
            gdesc1 = alpha * gdesc1 + (1.0 - alpha) * ldesc1
        # 5. Calculate scores based on global descriptors
        S = self.calculate_matching_score(gdesc0, gdesc1) * self.config['descriptor_dim'] ** -0.5

        # 6. Run optimal transport to get matching probabilities
        log_P = self.get_matching_probs(S)
        return {
            'context_descriptors0': gdesc0,
            'context_descriptors1': gdesc1,
            'scores': log_P
        }

    @staticmethod
    def normalize_keypoints(kpts, image_shape):
        """Normalize keypoints coordinates to the range [-1., 1.]"""
        height, width = image_shape[-2], image_shape[-1]
        return 2 * kpts / torch.tensor([height - 1, width - 1], device=kpts.device) - 1.

    @staticmethod
    def calculate_matching_score(desc0, desc1):
        """
        desc0 - [B, E, M]
        desc1 - [B, E, N]
        """
        return torch.matmul(desc0.transpose(1, 2).contiguous(), desc1)

    def get_matching_probs(self, S):
        """sinkhorn"""
        batch_size, m, n = S.size()
        # augment scores matrix
        S_aug = torch.empty(batch_size, m + 1, n + 1, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        S_aug[:, m, :] = self.dustbin_score
        S_aug[:, :, n] = self.dustbin_score

        # prepare normalized source and target log-weights
        norm = -torch.tensor(n + m, device=S.device).log()
        log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n + 1).contiguous()
        log_a[-1] += math.log(n)
        log_b[-1] += math.log(m)

        log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
        log_P = log_otp_solver(
            log_a,
            log_b,
            S_aug,
            num_iters=self.config['otp']['num_iters'],
            reg=self.config['otp']['reg']
        )
        return log_P - norm

import pathlib
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import nms2d

from models.features.superpoint.utils import remove_borders, sample_desc_from_points, top_k_keypoints
from models.features.utils import min_stack

conv2d = lambda ch_in, ch_out: nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
bn = lambda ch_out: nn.BatchNorm2d(ch_out)


class SuperPointNet(nn.Module):
    r"""Implementation of SuperPoint Detector and Descriptor
    Original paper: https://arxiv.org/abs/1712.07629
    """

    def __init__(self, max_keypoints: int = -1, descriptor_dim: int = 256, nms_kernel: int = 9,
                 remove_borders_size: int = 4, keypoint_threshold: float = 0.0,
                 weights: Optional[Union[str, pathlib.Path]] = None):
        super().__init__()

        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim
        self.nms_kernel = nms_kernel
        self.remove_borders_size = remove_borders_size
        self.keypoint_threshold = keypoint_threshold

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layers_channels = [
            [1, 64, 64, 64],
            [64, 64, 64, 64],
            [64, 128, 128, 128],
            [128, 128, 128, 128],
        ]

        for i, channels in enumerate(self.layers_channels):
            setattr(self, "conv{}a".format(i + 1), conv2d(channels[0], channels[1]))
            setattr(self, "conv{}b".format(i + 1), conv2d(channels[2], channels[3]))

        # Detector head
        self.convPa = conv2d(128, 256)  # 8 * 2
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor head
        self.convDa = conv2d(128, 256)
        self.convDb = nn.Conv2d(256, descriptor_dim, kernel_size=1, stride=1, padding=0)

        self._load_weights(weights)

    def _load_weights(self, weights: Optional[Union[str, pathlib.Path]] = None):
        if weights is not None:
            state_dict = torch.load(str(weights), map_location='cpu')
            message = self.load_state_dict(state_dict, strict=True)
            print(message)

    def _forward_layers(self, image: torch.Tensor, mask=None):
        """Apply SuperPoint layers"""
        x = image
        for i in range(4):
            x = self.relu(getattr(self, "conv{}a".format(i + 1))(x))
            x = self.relu(getattr(self, "conv{}b".format(i + 1))(x))
            if i != 3:
                x = self.pool(x)

        # Descriptors
        descriptors = self.convDb(self.relu(self.convDa(x)))
        dn = torch.norm(descriptors, p=2, dim=1)
        descriptors = descriptors.div(torch.unsqueeze(dn, 1))

        # Keypoints scores
        scores = self.convPb(self.relu(self.convPa(x)))
        scores = F.softmax(scores, 1)[:, :-1]
        return descriptors, scores

    def forward(self, image: torch.Tensor, mask=None):
        """Compute keypoints, scores, descriptors for image"""
        # Encoder

        descriptors, scores = self._forward_layers(image, mask)

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Applying non-maximum suppression for filtering keypoints scores
        scores = nms2d(scores.unsqueeze(1), kernel_size=(self.nms_kernel, self.nms_kernel)).squeeze(1)
        keypoints = [
            torch.nonzero(F.threshold(s, self.keypoint_threshold, 0.))
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Filter top-k keypoints, and remove those close to the borders
        keypoints, scores = list(
            zip(
                *[
                    top_k_keypoints(
                        remove_borders(k, s, self.remove_borders_size, h * 8, w * 8),
                        self.max_keypoints
                    )
                    for k, s in zip(keypoints, scores)
                ]
            )
        )
        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        descriptors = [
            sample_desc_from_points(k[None], d[None], 8)[0].transpose(0, 1)
            for k, d in zip(keypoints, descriptors)
        ]

        # Downscale to the same number of keypoints for each image to fit in a batch
        keypoints, scores, descriptors = min_stack(keypoints, scores, descriptors)

        # Converting keypoints to lafs
        # Set scale to 1
        lafs = torch.cat([
            torch.eye(2, device=keypoints.device, dtype=keypoints.dtype).unsqueeze(0).unsqueeze(1).expand(
                keypoints.size(0), keypoints.size(1), -1, -1
            ),
            keypoints.unsqueeze(-1),
        ], dim=-1)

        return lafs, scores, descriptors


class SuperPointNetBn(SuperPointNet):
    def __init__(self, max_keypoints: int = -1, descriptor_dim: int = 256, nms_kernel: int = 9,
                 remove_borders_size: int = 4, keypoint_threshold: float = 0.0,
                 weights: Optional[Union[str, pathlib.Path]] = None):

        super().__init__(max_keypoints, descriptor_dim, nms_kernel, remove_borders_size,
                         keypoint_threshold, weights=None)

        # Add batch norm layers
        for i, channels in enumerate(self.layers_channels):
            setattr(self, "bn{}a".format(i + 1), bn(channels[1]))
            setattr(self, "bn{}b".format(i + 1), bn(channels[3]))
        self.bnPa = bn(256)
        self.bnPb = bn(65)
        self.bnDa = bn(256)
        self.bnDb = bn(256)

        self._load_weights(weights)

    @staticmethod
    def rename_weights_keys(state_dict):
        for key in list(state_dict.keys()):
            state_dict[key.replace(
                'inc.conv.conv.0', 'conv1a').replace(
                'inc.conv.conv.1', 'bn1a').replace(
                'inc.conv.conv.3', 'conv1b').replace(
                'inc.conv.conv.4', 'bn1b').replace(
                'down1.mpconv.1.conv.0', 'conv2a').replace(
                'down1.mpconv.1.conv.1', 'bn2a').replace(
                'down1.mpconv.1.conv.3', 'conv2b').replace(
                'down1.mpconv.1.conv.4', 'bn2b').replace(
                'down2.mpconv.1.conv.0', 'conv3a').replace(
                'down2.mpconv.1.conv.1', 'bn3a').replace(
                'down2.mpconv.1.conv.3', 'conv3b').replace(
                'down2.mpconv.1.conv.4', 'bn3b').replace(
                'down3.mpconv.1.conv.0', 'conv4a').replace(
                'down3.mpconv.1.conv.1', 'bn4a').replace(
                'down3.mpconv.1.conv.3', 'conv4b').replace(
                'down3.mpconv.1.conv.4', 'bn4b')] = state_dict.pop(key)
        return state_dict

    def _load_weights(self, weights: Optional[Union[str, pathlib.Path]] = None):
        if weights is not None:
            state_dict = torch.load(str(weights), map_location='cpu')['model_state_dict']
            state_dict = self.rename_weights_keys(state_dict)
            message = self.load_state_dict(state_dict, strict=True)
            print(message)

    def _forward_layers(self, image: torch.Tensor, mask=None):
        """Apply SuperPoint layers"""
        x = image
        for i in range(4):
            x = self.relu(getattr(self, "bn{}a".format(i + 1))(getattr(self, "conv{}a".format(i + 1))(x)))
            x = self.relu(getattr(self, "bn{}b".format(i + 1))(getattr(self, "conv{}b".format(i + 1))(x)))
            if i != 3:
                x = self.pool(x)

        # Descriptors
        cDa = self.relu(self.bnDa(self.convDa(x)))
        descriptors = self.bnDb(self.convDb(cDa))
        dn = torch.norm(descriptors, p=2, dim=1)
        descriptors = descriptors.div(torch.unsqueeze(dn, 1))

        # Keypoints scores
        cPa = self.relu(self.bnPa(self.convPa(x)))
        scores = self.bnPb(self.convPb(cPa))
        scores = F.softmax(scores, 1)[:, :-1]
        return descriptors, scores

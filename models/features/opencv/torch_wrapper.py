from typing import Tuple

import kornia as K
import numpy as np
import torch
import torch.nn as nn

from .base import OpenCVFeatures


class OpenCVFeaturesTorchWrapper(nn.Module):
    """
    This class wraps opencv based feature extract of type OpenCVFeatures.
    Provides unified interface for all feature extractors that work with torch tensors and input and output.
    """

    def __init__(self, opencv_features: OpenCVFeatures):
        super(OpenCVFeaturesTorchWrapper, self).__init__()
        self.features = opencv_features

    def forward(self, image: torch.Tensor, mask=None) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Wrapper for OpenCV based feature extraction for torch Tensors.
        Output will be transferred to the same device as input.
        Args:
            image: tensor representation (B=1,1,H,W) of image. It is not recommended passing tensors on gpu as it will
            involve transferring data back and forth to cpu
            mask:

        Returns:
            Returns:
            lafs: tensor (B=1, N, 2, 3) of local affine frames created from detected keypoints
            scores: tensor (B=1,N,) of corresponding detector responses
            descriptors: tensor (B=1, N, 128) of descriptors
        """
        B, C, H, W = image.size()
        device = image.device

        # currently, support only batch size = 1
        assert B == 1
        image = (255. * K.tensor_to_image(image)).astype(np.uint8)
        lafs, scores, descriptors = map(
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
            self.features.detect_and_compute(image)
        )
        return lafs, scores, descriptors

    def __repr__(self):
        return f'OpenCVFeaturesTorchWrapper(features={self.features})'

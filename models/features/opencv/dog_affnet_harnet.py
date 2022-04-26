"""
This module contains mixed feature extractors, where part of the pipeline is done using OpenCV methods on CPU
and other part is done using torch-based modules either on CPU or GPU.
"""

from typing import Union, Tuple

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import torch.nn as nn
from kornia_moons.feature import laf_from_opencv_SIFT_kpts

from .base import detect_kpts_opencv


class DoGOpenCVAffNetHardNet(nn.Module):
    """
    Class for detecting features using OpenCV implementation of SIFT,
    affine-shape estimation with torch-based implementation of AffNet,
    and description with torch-based implementation of HardNet
    """

    def __init__(self, max_keypoints: int = -1, nms_diameter: float = 9.):
        super(DoGOpenCVAffNetHardNet, self).__init__()
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_diameter / 2

        self.features = cv2.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000)
        self.hardnet = KF.HardNet(True).eval()
        # Affine shape estimator
        self.affnet = KF.LAFAffNetShapeEstimator(True).eval()
        self.orinet = KF.LAFOrienter(32, angle_detector=KF.OriNet(True)).eval()

    @torch.no_grad()
    def detect_and_compute(self,
                           image: Union[np.ndarray, torch.Tensor],
                           mask=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect keypoint with DoG detector and apply HardNet description
        with AffNet for local affine shape estimation.
        Args:
            image: either numpy array (H, W) representing grayscale uint8 image or torch.Tensor (B=1, 1, H, W)
            of float data type normalized to [0., 1.] range

        Returns:

        """
        # keep both numpy uint8 array for image and float torch.Tensor
        if isinstance(image, torch.Tensor):
            # support only batch size = 1
            assert image.size(0) == 1
            device = image.device
            image_tensor = image
            image = (K.tensor_to_image(image_tensor) * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            image_tensor = K.image_to_tensor(image, keepdim=False).float() / 255.  # (1,1,H,W)
            device = torch.device('cpu')
        else:
            raise TypeError(f'Invalid type of `image` parameter: {type(image)}')

        kpts, scores = detect_kpts_opencv(self.features, image, self.nms_radius, self.max_keypoints)
        scores = torch.from_numpy(scores).unsqueeze(0).to(device)

        # convert OpenCV kpts to lafs in tensor format
        lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)  # (1,N,2,3)

        lafs = self.orinet(self.affnet(lafs, image_tensor), image_tensor)

        patches = KF.extract_patches_from_pyramid(image_tensor, lafs, PS=32)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit
        descriptors = self.hardnet(patches.view(B * N, CH, H, W)).view(B, N, -1)

        return lafs, scores, descriptors

    def forward(self, image: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.detect_and_compute(image, mask)

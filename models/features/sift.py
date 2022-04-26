from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature import LAFDescriptor
from kornia.feature.orientation import LAFOrienter, PassLAF
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid
from kornia.feature.responses import BlobDoG, CornerGFTT
from kornia.feature.siftdesc import SIFTDescriptor


import torch


from .base import Features


class SIFT(Features):
    r"""Module for computing SIFT descriptors
        Uses DoG detector + (Root)SIFT descriptor.
        For more details refer to https://kornia.readthedocs.io/en/latest/_modules/kornia/feature/integrated.html#SIFTFeature

        Args:
            config: Dict with optional parameters:
                'max_keypoints': Number of features
                'upright': bool
                'rootsift': bool, whether to compute RootSIFT (Arandjelović et. al, 2012)
    """

    def __init__(self, descriptor_dim=128, max_keypoints=8000, perform_nms=True, nms_diameter=9, patch_size=41, upright=False,
                 rootsift=True, device: torch.device = torch.device('cpu')):
        super(Features, self).__init__()
        self.detector = None
        self.nms_diameter = nms_diameter
        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim

        self.perform_nms = perform_nms

        self.detector = ScaleSpaceDetector(self.max_keypoints,
                                      resp_module=BlobDoG(),
                                      nms_module=ConvQuadInterp3d(10),
                                      scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
                                      ori_module=PassLAF() if upright else LAFOrienter(19),
                                      scale_space_response=True,
                                      minima_are_also_good=True,
                                      mr_size=6.0).to(device)

        self.descriptor = LAFDescriptor(SIFTDescriptor(patch_size=patch_size, rootsift=rootsift),
                                   patch_size=patch_size,
                                   grayscale_descriptor=True).to(device)

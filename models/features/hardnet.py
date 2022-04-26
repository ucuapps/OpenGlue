from kornia.feature import LAFDescriptor
from kornia.feature.affine_shape import LAFAffNetShapeEstimator
from kornia.feature.orientation import LAFOrienter, PassLAF
from kornia.feature.responses import CornerGFTT
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid


import torch


from .base import Features


class GFTTAffNetHardNet(Features):
    r"""Module for computing GFTTAffNetHardNet descriptors
    """

    def __init__(self, descriptor_dim=128, max_keypoints=8000, perform_nms=True, nms_diameter=9, patch_size=32,
                 upright=False, device: torch.device = torch.device('cpu')):
        super(Features, self).__init__()
        self.perform_nms = perform_nms
        self.nms_diameter = nms_diameter
        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim

        self.detector = ScaleSpaceDetector(self.max_keypoints,
                                           resp_module=CornerGFTT(),
                                           nms_module=ConvQuadInterp3d(10, 1e-5),
                                           scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=False),
                                           ori_module=PassLAF() if upright else LAFOrienter(19),
                                           aff_module=LAFAffNetShapeEstimator(True).eval(),
                                           mr_size=6.0).to(device)

        self.descriptor = LAFDescriptor(None,
                                        patch_size=patch_size,
                                        grayscale_descriptor=True).to(device)

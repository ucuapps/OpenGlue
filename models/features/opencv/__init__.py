from ._features_torch import sift_create_torch
from .dog_affnet_harnet import DoGOpenCVAffNetHardNet

methods = {
    'OPENCV_SIFT': sift_create_torch,
    'OPENCVDoGAffNetHardNet': DoGOpenCVAffNetHardNet
    # register new methods here
}

__all__ = ['methods']

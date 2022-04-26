"""
Convenience functions for creating several types of OpenCV based descriptors wrapped with OpenCVFeaturesTorchWrapper.
"""

from ._features import sift_create
from .torch_wrapper import OpenCVFeaturesTorchWrapper


def sift_create_torch(max_keypoints: int = -1, nms_diameter: float = 9.,
                      rootsift: bool = True) -> OpenCVFeaturesTorchWrapper:
    return OpenCVFeaturesTorchWrapper(
        sift_create(max_keypoints, nms_diameter, rootsift)
    )

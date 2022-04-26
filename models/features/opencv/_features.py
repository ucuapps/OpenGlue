"""
Convenience functions for creating several types of OpenCV based descriptors.
"""

import cv2

from .base import OpenCVFeatures


def sift_create(max_keypoints: int = -1, nms_diameter: float = 9., rootsift: bool = True) -> OpenCVFeatures:
    return OpenCVFeatures(
        cv2.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000),
        max_keypoints=max_keypoints,
        nms_diameter=nms_diameter,
        normalize_desc=True,
        root_norm=rootsift,
        laf_scale_mr_size=6.0
    )

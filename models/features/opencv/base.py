"""
Module with class that wraps OpenCV based detectors and descriptors.
Allows performing Non-Maximum suppression based on keypoints response, top-response keypoints filtering,
descriptors normalization.
"""

from typing import Union, Iterable, Tuple, Optional

import cv2
import numpy as np
from scipy.spatial import KDTree


class OpenCVFeatures:
    def __init__(self, features: cv2.Feature2D, max_keypoints: int = -1,
                 nms_diameter: float = 9., normalize_desc: bool = True, root_norm: bool = True,
                 laf_scale_mr_size: Optional[float] = 6.0):
        self.features = features

        self.max_keypoints = max_keypoints
        self.nms_radius = nms_diameter / 2
        self.normalize_desc = normalize_desc
        self.root_norm = root_norm
        self.laf_scale_mr_size = laf_scale_mr_size

    @staticmethod
    def normalize_descriptors(descriptors: np.ndarray, root_norm: bool = True) -> np.ndarray:
        """
        Normalize descriptors.
        If root_norm=True apply RootSIFT-like normalization, else regular L2 normalization.
        Args:
            descriptors: array (N, 128) with unnormalized descriptors
            root_norm: boolean flag indicating whether to apply RootSIFT-like normalization

        Returns:
            descriptors: array (N, 128) with normalized descriptors
        """
        descriptors = descriptors.astype(np.float32)
        if root_norm:
            # L1 normalize
            norm = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
            descriptors /= norm
            # take square root of descriptors
            descriptors = np.sqrt(descriptors)
        else:
            # L2 normalize
            norm = np.linalg.norm(descriptors, ord=2, axis=1, keepdims=True)
            descriptors /= norm
        return descriptors

    @staticmethod
    def lafs_from_opencv_kpts(kpts: Iterable[cv2.KeyPoint],
                              mr_size: float = 6.0,
                              with_resp: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert OpenCV keypoint to Local Affine Frames.
        Adapted from kornia_moons for numpy arrays.
        https://github.com/ducha-aiki/kornia_moons/blob/6aa7bdbe1879303bd9bf35494b383e4f959a1135/kornia_moons/feature.py#L60

        Args:
            kpts: iterable of OpenCV keypoints
            mr_size: multiplier for keypoint size
            with_resp: flag indicating whether to return responses

        Returns:
            lafs: array (N, 2, 3) of local affine frames made from keypoints
            responses (optional): array (N,) of responses corresponding to lafs
        """

        xy = np.array([k.pt for k in kpts], dtype=np.float32)
        scales = np.array([mr_size * k.size for k in kpts], dtype=np.float32)
        angles = np.array([k.angle for k in kpts], dtype=np.float32)
        # if angles are not set, make them 0
        if np.allclose(angles, -1.):
            angles = np.zeros_like(scales, dtype=np.float32)
        angles = np.deg2rad(-angles)

        n = xy.shape[0]
        lafs = np.empty((n, 2, 3), dtype=np.float32)
        lafs[:, :, 2] = xy
        s_cos_t = scales * np.cos(angles)
        s_sin_t = scales * np.sin(angles)
        lafs[:, 0, 0] = s_cos_t
        lafs[:, 0, 1] = s_sin_t
        lafs[:, 1, 0] = -s_sin_t
        lafs[:, 1, 1] = s_cos_t

        if with_resp:
            resp = np.array([k.response for k in kpts], dtype=np.float32)
            return lafs, resp
        else:
            return lafs

    def detect_and_compute(self, image: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect keypoint with OpenCV-based detector and apply OpenCV-based description.
        Args:
            image: array representation of grayscale image of uint8 data type
        Returns:
            lafs: array (N, 2, 3) of local affine frames created from detected keypoints
            scores: array (N,) of corresponding detector responses
            descriptors: array (N, 128) of descriptors
        """
        kpts, scores, descriptors = detect_kpts_opencv(self.features, image, self.nms_radius, self.max_keypoints,
                                                       describe=True)
        lafs = self.lafs_from_opencv_kpts(kpts, mr_size=self.laf_scale_mr_size, with_resp=False)

        if self.normalize_desc:
            descriptors = self.normalize_descriptors(descriptors, self.root_norm)

        return lafs, scores, descriptors

    def __repr__(self):
        return f'OpenCVFeatures(features={type(self.features)})'


def detect_kpts_opencv(features: cv2.Feature2D, image: np.ndarray, nms_radius: float, max_keypoints: int,
                       describe: bool = False) -> np.ndarray:
    """
    Detect keypoints using OpenCV Detector. Optionally, perform NMS and filter top-response keypoints.
    Optionally perform description.
    Args:
        features: OpenCV based keypoints detector and descriptor
        image: Grayscale image of uint8 data type
        nms_radius: radius of non-maximum suppression. If negative, skip nms
        max_keypoints: maximum number of keypoints to keep based on response. If negative, keep all
        describe: flag indicating whether to simultaneously compute descriptors

    Returns:
        kpts: 1D array of detected cv2.KeyPoint
    """
    if describe:
        kpts, descriptors = features.detectAndCompute(image, None)
    else:
        kpts = features.detect(image, None)
    kpts = np.array(kpts)

    responses = np.array([k.response for k in kpts], dtype=np.float32)
    kpts_pt = np.array([k.pt for k in kpts], dtype=np.float32)

    if nms_radius > 0:
        nms_mask = nms_keypoints(kpts_pt, responses, nms_radius)
    else:
        nms_mask = np.ones((kpts_pt.shape[0],), dtype=bool)

    responses = responses[nms_mask]
    kpts = kpts[nms_mask]

    if max_keypoints > 0:
        top_score_idx = np.argpartition(-responses, min(max_keypoints, len(responses) - 1))[:max_keypoints]
    else:
        # select all
        top_score_idx = ...

    if describe:
        return kpts[top_score_idx], responses[top_score_idx], descriptors[nms_mask][top_score_idx]
    else:
        return kpts[top_score_idx], responses[top_score_idx]


def nms_keypoints(kpts: np.ndarray, responses: np.ndarray, radius: float) -> np.ndarray:
    # TODO: add approximate tree
    kd_tree = KDTree(kpts)

    sorted_idx = np.argsort(-responses)
    kpts_to_keep_idx = []
    removed_idx = set()

    for idx in sorted_idx:
        # skip point if it was already removed
        if idx in removed_idx:
            continue

        kpts_to_keep_idx.append(idx)
        point = kpts[idx]
        neighbors = kd_tree.query_ball_point(point, r=radius)
        # Variable `neighbors` contains the `point` itself
        removed_idx.update(neighbors)

    mask = np.zeros((kpts.shape[0],), dtype=bool)
    mask[kpts_to_keep_idx] = True
    return mask

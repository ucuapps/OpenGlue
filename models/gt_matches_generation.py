"""
Module for generating ground truth matches between two images given keypoints on both images
and ground truth transformation
"""

from typing import Dict, Any, Optional, Tuple

import torch

from utils.misc import get_inverse_transformation, reproject_keypoints

# define module constants
UNMATCHED_INDEX = -1  # index of keypoint that don't have a match
IGNORE_INDEX = -2  # index of keypoints to ignore during loss calculation


def generate_gt_matches(data: Dict[str, Any],
                        features0: Dict[str, torch.Tensor],
                        features1: Dict[str, torch.Tensor],
                        positive_threshold: float,
                        negative_threshold: Optional[float] = None
                        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, torch.Tensor]]]:
    """Given image pair, keypoints detected in each image, return set of ground truth correspondences"""
    if negative_threshold is None:
        negative_threshold = positive_threshold

    kpts0, kpts1 = features0['keypoints'], features1['keypoints']

    transformation = data['transformation']
    transformation_inv = get_inverse_transformation(transformation)
    num0, num1 = kpts0.size(1), kpts1.size(1)

    # skip step if no keypoint are detected
    if num0 == 0 or num1 == 0:
        return None, None

    # establish ground truth correspondences given transformation
    kpts0_transformed, mask0 = reproject_keypoints(kpts0, transformation)
    kpts1_transformed, mask1 = reproject_keypoints(kpts1, transformation_inv)
    reprojection_error_0_to_1 = torch.cdist(kpts0_transformed, kpts1, p=2)  # batch_size x num0 x num1
    reprojection_error_1_to_0 = torch.cdist(kpts1_transformed, kpts0, p=2)  # batch_size x num1 x num0

    min_dist0, nn_matches0 = reprojection_error_0_to_1.min(2)  # batch_size x num0
    min_dist1, nn_matches1 = reprojection_error_1_to_0.min(2)  # batch_size x num1
    gt_matches0, gt_matches1 = nn_matches0.clone(), nn_matches1.clone()
    device = gt_matches0.device
    cross_check_consistent0 = torch.arange(num0, device=device).unsqueeze(0) == gt_matches1.gather(1, gt_matches0)
    gt_matches0[~cross_check_consistent0] = UNMATCHED_INDEX

    cross_check_consistent1 = torch.arange(num1, device=device).unsqueeze(0) == gt_matches0.gather(1, gt_matches1)
    gt_matches1[~cross_check_consistent1] = UNMATCHED_INDEX

    # so far mutual NN are marked MATCHED and non-mutual UNMATCHED

    symmetric_dist = 0.5 * (min_dist0[cross_check_consistent0] + min_dist1[cross_check_consistent1])

    gt_matches0_cross = gt_matches0[cross_check_consistent0].clone()
    gt_matches1_cross = gt_matches1[cross_check_consistent1].clone()
    gt_matches0_uncross = gt_matches0[~cross_check_consistent0].clone()
    gt_matches1_uncross = gt_matches1[~cross_check_consistent1].clone()

    gt_matches0_cross[symmetric_dist > positive_threshold] = IGNORE_INDEX
    gt_matches0_cross[symmetric_dist > negative_threshold] = UNMATCHED_INDEX

    gt_matches1_cross[symmetric_dist > positive_threshold] = IGNORE_INDEX
    gt_matches1_cross[symmetric_dist > negative_threshold] = UNMATCHED_INDEX

    gt_matches0_uncross[min_dist0[~cross_check_consistent0] <= negative_threshold] = IGNORE_INDEX
    gt_matches1_uncross[min_dist1[~cross_check_consistent1] <= negative_threshold] = IGNORE_INDEX

    # mutual NN with sym_dist <= pos.th ==> MATCHED
    # mutual NN with  pos.th < sym_dist <= neg.th ==> IGNORED
    # mutual NN with neg.th < sym_dist => UNMATCHED
    # non-mutual with dist <= neg.th ==> IGNORED
    # non-mutual with dist > neg.th ==> UNMATCHED

    # ignore kpts with unknown depth data
    gt_matches0[~mask0] = IGNORE_INDEX
    gt_matches1[~mask1] = IGNORE_INDEX

    # also ignore MATCHED point if its nearest neighbor is invalid
    gt_matches0_cross[~mask1.gather(1, nn_matches0)[cross_check_consistent0]] = IGNORE_INDEX
    gt_matches1_cross[~mask0.gather(1, nn_matches1)[cross_check_consistent1]] = IGNORE_INDEX

    # update gt_matches0, gt_matches1
    gt_matches0.masked_scatter_(cross_check_consistent0, gt_matches0_cross)
    gt_matches1.masked_scatter_(cross_check_consistent1, gt_matches1_cross)


    data = {
        **data,
        'keypoints0': kpts0, 'keypoints1': kpts1,
        'local_descriptors0': features0['local_descriptors'], 'local_descriptors1': features1['local_descriptors'],
        'side_info0': features0['side_info'], 'side_info1': features1['side_info'],
    }

    y_true = {
        'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1
    }

    return data, y_true

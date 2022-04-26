import inspect

import kornia.feature as KF
import numpy as np
import torch


def filter_dict(dict_to_filter, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys}
    return filtered_dict


def get_descriptors(img, descriptor, lafs=None, patch_size=32):
    r"""Acquire descriptors for each keypoint given an original image, its keypoints, and a descriptor module"""

    patches = KF.extract_patches_from_pyramid(img, lafs, patch_size)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

    return descs


def min_stack(keypoints, side_info, descriptors):
    """
    Stack batch of keypoints prediction into single tensor.
    For each instance keep number of keypoints minimal in the batch.
    """
    kpts_num = np.array([x.shape[0] for x in keypoints])
    min_kpts_to_keep = kpts_num.min()

    if np.all(kpts_num == min_kpts_to_keep):
        return torch.stack(keypoints), torch.stack(side_info), torch.stack(descriptors)
    else:
        # get scores and indices of keypoints to keep in each batch element
        indices_to_keep = [torch.topk(side_info_, min_kpts_to_keep, dim=0).indices
                           for side_info_ in side_info]

        data_stacked = {'keypoints': [], 'side_info': [], 'descriptors': []}
        for kpts, descs, info, idxs in zip(keypoints, descriptors, side_info, indices_to_keep):
            data_stacked['side_info'].append(info[idxs])
            data_stacked['keypoints'].append(kpts[idxs])
            data_stacked['descriptors'].append(descs[idxs])

        keypoints = torch.stack(data_stacked['keypoints'])
        side_info = torch.stack(data_stacked['side_info'])
        descriptors = torch.stack(data_stacked['descriptors'])

        return keypoints, side_info, descriptors


def prepare_features_output(lafs, responses, desc, laf_converter, permute_desc=False, log_response=False):
    """Convert features output into format acceptable by SuperGlue"""
    kpts = lafs[:, :, :, -1]
    responses = responses.unsqueeze(-1)
    if log_response:
        responses = (responses + 0.1).log()
    side_info = torch.cat([responses, laf_converter(lafs)], dim=-1)
    return {
        'keypoints': kpts,
        'side_info': side_info,
        'local_descriptors': desc.permute(0, 2, 1) if permute_desc else desc
    }
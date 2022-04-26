from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.subpix import nms2d


class Features(nn.Module):

    def __init__(self, detector, descriptor, perform_nms: bool = True, nms_diameter: int = 9, max_keypoints: int = 1024):
        super(Features, self).__init__()
        self.detector = detector
        self.descriptor = descriptor
        self.perform_nms = perform_nms
        self.nms_diameter = nms_diameter
        self.max_keypoints = max_keypoints

    @torch.no_grad()
    def run_nms(self, lafs, scores, img_size):
        """Perform non-maxima suppression over detected keypoints"""
        device = lafs.device
        b, _, h, w = img_size

        lafs_list, scores_list = [], []
        mask = torch.zeros(1, 1, h, w, device=device)

        for i, (lafs_, scores_) in enumerate(zip(lafs, scores)):
            kpts_ = torch.round(lafs_[:, :, -1]).type(torch.long)
            scores_min = scores_.amin()
            scores_max = scores_.amax()

            # monotonic function of scores_ in range (0, 1)
            normalized_scores = (scores_ - scores_min + 1e-2) / (scores_max - scores_min + 1e-1)
            kpt_value_xy = h * kpts_[:, 0] + kpts_[:, 1]
            kpt_value_xyr = kpt_value_xy + (1 - normalized_scores)
            kpts_sorted_idx = torch.argsort(kpt_value_xyr)

            kpt_value_xy = kpt_value_xy[kpts_sorted_idx]
            # get indices of unique kpts locations with highest response
            _, unique_idx = torch.unique_consecutive(kpt_value_xy, return_inverse=True)
            unique_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=device),
                 torch.where(unique_idx[1:] - unique_idx[:-1])[0] + 1]
            )
            unique_idx = kpts_sorted_idx[unique_idx]

            kpts_ = kpts_[unique_idx]
            scores_ = scores_[unique_idx]
            mask[0, 0, kpts_[:, 1], kpts_[:, 0]] = scores_

            # NMS
            nms_mask = nms2d(mask, kernel_size=(self.nms_diameter, self.nms_diameter), mask_only=True)
            nms_mask = nms_mask[0, 0, kpts_[:, 1], kpts_[:, 0]]

            scores_ = scores_[nms_mask]
            lafs_ = lafs_[unique_idx][nms_mask]
            scores_list.append(scores_)
            lafs_list.append(lafs_)
            mask.zero_()

        # min stack
        kpts_num = np.array([x.shape[0] for x in lafs_list]).min()
        kpts_num = min(kpts_num, self.max_keypoints)

        # get scores and indices of keypoints to keep in each batch element
        indices_to_keep = [torch.topk(scores_, kpts_num, dim=0).indices
                           for scores_ in scores_list]
        for i, idx in enumerate(indices_to_keep):
            lafs_list[i] = lafs_list[i][idx]
            scores_list[i] = scores_list[i][idx]

        return torch.stack(lafs_list, dim=0), torch.stack(scores_list, dim=0)

    def forward(self, image: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lafs, scores = self.detector(image)

        if self.perform_nms:
            lafs, scores = self.run_nms(lafs, scores, image.size())

        descriptors = self.descriptor(image, lafs)

        return lafs, scores, descriptors
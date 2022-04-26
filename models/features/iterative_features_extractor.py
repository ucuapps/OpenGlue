from typing import Tuple, Optional

import torch
import torch.nn as nn

from .utils import min_stack


class IterativeLocalFeature(nn.Module):
    """Iteratively extract features for each pair in the batch.
    Convenience module for local features extractors that don't support batching
    Applies min-stack for batching"""

    def __init__(self, features_extractor_builder, *args, **kwargs):
        super(IterativeLocalFeature, self).__init__()
        self.features_extractor = features_extractor_builder(*args, **kwargs)

    def forward(self,
                img: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                              torch.Tensor,
                                                              torch.Tensor]:
        batch_size = img.size(0)
        lafs_list, resp_list, desc_list = [], [], []
        for idx in range(batch_size):
            lafs, resp, desc = self.features_extractor(img[idx:idx + 1],
                                                       mask[idx: idx + 1] if mask is not None else None)
            lafs_list.append(lafs[0])
            resp_list.append(resp[0])
            desc_list.append(desc[0])

        lafs, resp, desc = min_stack(lafs_list, resp_list, desc_list)
        return lafs, resp, desc

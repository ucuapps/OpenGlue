import torch


def remove_borders(keypoints, scores, border_size: int, H: int, W: int):
    """Removing keypoints near the border"""
    toremoveW = torch.logical_or(keypoints[:, 1] < border_size, keypoints[:, 1] >= (W - border_size))
    toremoveH = torch.logical_or(keypoints[:, 0] < border_size, keypoints[:, 0] >= (H - border_size))
    toremove = torch.logical_or(toremoveW, toremoveH)

    return keypoints[~toremove, :], scores[~toremove]


def sample_desc_from_points(pts, coarse_desc, cell=8):
    """
    inputs:
        coarse_desc: tensor [1, 256, Hc, Wc]
        pts: tensor [N, 2] (should be the same device as desc)
    return:
        desc: tensor [N, D]
    """
    D, H, W = coarse_desc.shape[1], coarse_desc.shape[2] * cell, coarse_desc.shape[3] * cell

    pts = pts - cell / 2 + 0.5
    pts /= torch.tensor([(W - cell / 2 - 0.5), (H - cell / 2 - 0.5)]).to(pts.device)
    pts = pts.view(1, 1, -1, 2)
    pts = pts * 2 - 1

    desc = torch.nn.functional.grid_sample(coarse_desc, pts, align_corners=False).view(1, D, -1)
    desc = torch.nn.functional.normalize(desc, p=2, dim=1)

    return desc


def top_k_keypoints(x, k: int):
    keypoints, scores = x
    if k >= len(keypoints) or k == -1:
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

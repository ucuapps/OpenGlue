import torch
import torch.nn.functional as F


def normalize_with_intrinsics(kpts: torch.tensor, K: torch.tensor):
    kpts0_calibrated = (kpts - K[:2, 2].unsqueeze(0)) / K[[0, 1], [0, 1]].unsqueeze(0)
    return kpts0_calibrated


def data_to_device(data, device):
    """Recursively transfers all tensors in dictionary to given device"""
    if isinstance(data, torch.Tensor):
        res = data.to(device)
    elif isinstance(data, dict):
        res = {k: data_to_device(v, device) for k, v in data.items()}
    else:
        res = data
    return res


def reproject_keypoints(kpts, transformation):
    """Reproject batch of keypoints given corresponding transformations"""
    transformation_type = transformation['type'][0]
    if transformation_type == 'perspective':
        H = transformation['H']
        return perspective_transform(kpts, H)
    elif transformation_type == '3d_reprojection':
        K0, K1 = transformation['K0'], transformation['K1']
        T, R = transformation['T'], transformation['R']
        depth0 = transformation['depth0']
        return reproject_3d(kpts, K0, K1, T, R, depth0)
    else:
        raise ValueError(f'Unknown transformation type {transformation_type}.')


def get_inverse_transformation(transformation):
    transformation_type = transformation['type'][0]
    if transformation_type == 'perspective':
        H = transformation['H']
        return {
            'type': transformation['type'],
            'H': torch.linalg.inv(H)
        }
    elif transformation_type == '3d_reprojection':
        K0, K1 = transformation['K0'], transformation['K1']
        T, R = transformation['T'], transformation['R']
        depth0, depth1 = transformation['depth0'], transformation['depth1']
        R_t = torch.transpose(R, 1, 2).contiguous()
        return {
            'type': transformation['type'],
            'K0': K1,
            'K1': K0,
            'R': R_t,
            'T': -torch.matmul(R_t, T.unsqueeze(-1)).squeeze(-1),
            'depth0': depth1,
            'depth1': depth0,
        }
    else:
        raise ValueError(f'Unknown transformation type {transformation_type}.')


def perspective_transform(kpts, H, eps=1e-8):
    """Transform batch of keypoints given batch of homography matrices"""
    batch_size, num_kpts, _ = kpts.size()
    kpts = torch.cat([kpts, torch.ones(batch_size, num_kpts, 1, device=kpts.device)], dim=2)
    Ht = torch.transpose(H, 1, 2).contiguous()
    kpts_transformed = torch.matmul(kpts, Ht)
    kpts_transformed = kpts_transformed[..., :2] / (kpts_transformed[..., 2].unsqueeze(-1) + eps)
    mask = torch.ones(batch_size, num_kpts, dtype=torch.bool, device=kpts.device)  # all keypoints are valid
    return kpts_transformed, mask


def reproject_3d(kpts, K0, K1, T, R, depth0, eps=1e-8):
    """Transform batch of keypoints given batch of relative poses and depth maps"""
    batch_size, num_kpts, _ = kpts.size()
    kpts_hom = torch.cat([kpts, torch.ones(batch_size, num_kpts, 1, device=kpts.device)], dim=2)

    K0_inv = torch.linalg.inv(K0)
    K0_inv_t = torch.transpose(K0_inv, 1, 2).contiguous()
    K1_t = torch.transpose(K1, 1, 2).contiguous()
    R_t = torch.transpose(R, 1, 2).contiguous()

    # transform to ray space
    kpts_transformed = torch.matmul(kpts_hom, K0_inv_t)

    # if depth is in image format, len(depth0.shape) > 2 than select depth at kpts locations
    if len(depth0.shape) == 2:
        depth = depth0
    else:
        depth_idx = kpts.type(torch.int64)
        depth = depth0[
            torch.arange(batch_size, device=kpts.device).unsqueeze(-1),
            depth_idx[..., 1],
            depth_idx[..., 0]
        ]
    mask = ~torch.isclose(depth, depth.new_tensor(0.0))  # mask for values with missing depth information
    # multiply by corresponding depth
    kpts_transformed = kpts_transformed * depth.unsqueeze(-1)
    # apply (R, T)
    kpts_transformed = torch.matmul(kpts_transformed, R_t) + T.unsqueeze(1)
    kpts_transformed = torch.matmul(kpts_transformed, K1_t)
    kpts_transformed = kpts_transformed[..., :2] / (kpts_transformed[..., 2].unsqueeze(-1) + eps)
    return kpts_transformed, mask


def pairwise_cosine_dist(x1, x2):
    """
    Return pairwise half of cosine distance in range [0, 1].
    dist = (1 - cos(theta)) / 2
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return 0.25 * torch.cdist(x1, x2).pow(2)


def arange_like(x, dim: int):
    return torch.arange(x.shape[dim], dtype=x.dtype, device=x.device)

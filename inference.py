import cv2
import os
from omegaconf import OmegaConf
from typing import Dict, Optional
import argparse
import matplotlib.pyplot as plt

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

import torch
import torch.nn as nn

from models.features import get_feature_extractor
from models.features.utils import prepare_features_output
from models.laf_converter import get_laf_to_sideinfo_converter
from models.superglue.superglue import SuperGlue
from utils.misc import arange_like


def load_torch_image(fname, resize_to=None, device=torch.device('cpu')):
    image = cv2.imread(fname)
    timg = K.color.bgr_to_grayscale(K.image_to_tensor(image, False) / 255.).to(device)
    if resize_to is not None:
        new_w, new_h = resize_to
        timg = K.geometry.resize(timg, (new_h, new_w))
    return timg


def preds_to_device(preds, device):
    lafs, resp, desc = preds

    lafs, resp, desc = (torch.tensor(lafs).unsqueeze(0).to(device),
                        torch.tensor(resp).unsqueeze(0).to(device),
                        torch.tensor(desc).unsqueeze(0).to(device))
    return lafs, resp, desc


@torch.no_grad()
def initialize_models(experiment_path,
                      checkpoint_name,
                      device=torch.device('cpu'),
                      max_features=2048,
                      resize_to='original'):
    config_path = os.path.join(experiment_path, 'config.yaml')
    config = OmegaConf.load(config_path)

    features_config = OmegaConf.load(os.path.join(config['data']['root_path'],
                                                  config['data']['features_dir'], 'config.yaml'))
    config['features'] = features_config

    checkpoint_path = os.path.join(experiment_path, checkpoint_name)
    config['features']['max_keypoints'] = max_features
    if isinstance(resize_to, str):
        assert resize_to in ["as in config", "original"]
        if resize_to == 'original':
            config['data']['target_size'] = None

    else:
        assert len(resize_to) == 2
        assert resize_to[0] > 0
        assert resize_to[1] > 0
        config['data']['target_size'] = resize_to
    print(config['features'])

    # Initialize models & load weights
    local_features_extractor = get_feature_extractor(config['features']['name'])(**config['features']['parameters'])
    local_features_extractor.to(device)

    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('superglue.', '')] = state_dict.pop(key)
    superglue = SuperGlue(config['superglue'])
    message = superglue.load_state_dict(state_dict)
    print(message)
    superglue.to(device)
    return superglue, local_features_extractor, config


class OpenGlueMatcher(nn.Module):
    r"""Module, which finds correspondences between two images based on local features,
    followed with SuperGlue model.
    Args:
        local_feature: Local feature detector. See :class:`~kornia.feature.GFTTAffNetHardNet`.
        matcher: SuperGlue matcher, see :class:`~kornia.feature.DescriptorMatcher`.
    Returns:
        Dict[str, torch.Tensor]: Dictionary with image correspondences and confidence scores.
    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> gftt_hardnet_matcher = LocalFeatureMatcher(
        ...     GFTTAffNetHardNet(10), kornia.feature.DescriptorMatcher('snn', 0.8)
        ... )
        >>> out = gftt_hardnet_matcher(input)
    """

    def __init__(self, local_feature: nn.Module, matcher: nn.Module, match_config: Dict = {}) -> None:
        super().__init__()
        self.local_feature = local_feature
        self.laf_converter = get_laf_to_sideinfo_converter(match_config['superglue']['laf_to_sideinfo_method'])
        self.matcher = matcher
        self.match_config = match_config
        self.eval()

    def extract_features(self,
                         image: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Function for feature extraction from simple image."""
        lafs0, resps0, descs0 = self.local_feature(image)
        return {"lafs": lafs0, "responses": resps0, "descriptors": descs0}

    def no_match_output(self, device: torch.device, dtype: torch.dtype) -> dict:
        return {
            'keypoints0': torch.empty(0, 2, device=device, dtype=dtype),
            'keypoints1': torch.empty(0, 2, device=device, dtype=dtype),
            'lafs0': torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            'lafs1': torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            'confidence': torch.empty(0, device=device, dtype=dtype),
            'batch_indexes': torch.empty(0, device=device, dtype=torch.long)
        }

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: dictionary containing the input data in the following format:
        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.
        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``lafs0``, matching LAFs from image0 :math:`(1, NC, 2, 3)`.
            - ``lafs1``, matching LAFs from image1 :math:`(1, NC, 2, 3)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        if ('lafs0' not in data.keys()) or ('descriptors0' not in data.keys()):
            # One can supply pre-extracted local features
            feats_dict0: Dict[str, torch.Tensor] = self.extract_features(data['image0'])
            lafs0, descs0, resps0 = feats_dict0['lafs'], feats_dict0['descriptors'], feats_dict0['responses']
        else:
            lafs0, descs0, resps0 = data['lafs0'], data['descriptors0'], data['responses0']

        if ('lafs1' not in data.keys()) or ('descriptors1' not in data.keys()):
            feats_dict1: Dict[str, torch.Tensor] = self.extract_features(data['image1'])
            lafs1, descs1, resps1 = feats_dict1['lafs'], feats_dict1['descriptors'], feats_dict1['responses']

        else:
            lafs1, descs1, resps1 = data['lafs1'], data['descriptors1'], data['responses1']

        # Here the magic happens
        b0, c0, h0, w0 = data['image0'].shape
        b1, c1, h1, w1 = data['image1'].shape

        data['image0_size'], data['image1_size'] = [w0, h0], [w1, h1]
        log_transform_response = self.match_config['superglue'].get('log_transform_response', False)
        features0 = prepare_features_output(lafs0, resps0, descs0, self.laf_converter,
                                            log_response=log_transform_response)
        features1 = prepare_features_output(lafs1, resps1, descs1, self.laf_converter,
                                            log_response=log_transform_response)

        pred = {**data, **{k + '0': v for k, v in features0.items()}}
        pred = {**pred, **{k + '1': v for k, v in features1.items()}}
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)) and isinstance(data[k][0], torch.Tensor):
                data[k] = torch.stack(data[k])

        # predict matching scores
        scores = self.matcher(data)['scores']

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        arange0 = arange_like(indices0, 1)[None].expand(b0, -1)
        arange1 = arange_like(indices1, 1)[None].expand(b1, -1)

        mutual0 = arange0 == indices1.gather(1, indices0)
        mutual1 = arange1 == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.match_config['inference']['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        mask0 = indices0 != -1
        batch_idxs0 = arange_like(data['image0'], 0).unsqueeze(1).expand(-1, indices0.shape[1]).long()

        matching_idxs0 = arange0[mask0]
        batch_idxs0 = batch_idxs0[mask0]
        matching_idxs1 = indices0[mask0]

        matching_idxs = torch.stack([matching_idxs0, matching_idxs1], dim=-1)
        confidence = mscores0[mask0]
        mlafs0 = lafs0[batch_idxs0, matching_idxs0][None]
        mlafs1 = lafs1[batch_idxs0, matching_idxs1][None]
        out = {"original_matching_idxs": matching_idxs,
               "batch_indexes": batch_idxs0,
               "confidence": confidence,
               "lafs0": mlafs0,
               "lafs1": mlafs1,
               "keypoints0": KF.get_laf_center(mlafs0)[0],
               "keypoints1": KF.get_laf_center(mlafs1)[0]}

        return out


def run_inference(image0_path, image1_path, experiment_path, checkpoint_name, device='cpu'):
    max_features = 2048  # as for the IMC track
    resize_to = 'original'  # we will not resize input images for our example

    matcher, feature_extractor, config = initialize_models(experiment_path,
                                                           checkpoint_name,
                                                           torch.device(device),
                                                           max_features,
                                                           resize_to)
    timg0 = load_torch_image(image0_path)
    timg1 = load_torch_image(image1_path)

    sg = OpenGlueMatcher(feature_extractor, matcher, config)
    with torch.no_grad():
        out = sg({"image0": timg0, "image1": timg1})

    F, inliers = cv2.findFundamentalMat(out['keypoints0'].detach().cpu().numpy(),
                                        out['keypoints1'].detach().cpu().numpy(),
                                        cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
    inliers = inliers > 0

    return timg0, timg1, out['lafs0'], out['lafs1'], inliers


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--image0_path', type=str, help='path to image file')
    parser.add_argument('--image1_path', type=str, help='path to second image file')
    parser.add_argument('--experiment_path', type=str, help='path to directory with saved experiment that contains '
                                                            'checkpoints')
    parser.add_argument('--checkpoint_name', type=str, help='name of checkpoint weight to use at inference')
    parser.add_argument('--output_dir', type=str, help='path to a resulting image with matched points visualized',
                        default='result.png')
    parser.add_argument('--device', type=str, help='device to use for inference', default='cpu')

    args = parser.parse_args()

    # For consistency, config should be taken directly from the trained experiment directory
    img0, img1, lafs0, lafs1, inliers = run_inference(args.image0_path, args.image1_path, args.experiment_path,
                                                      args.checkpoint_name, args.device)

    draw_LAF_matches(
        lafs0,
        lafs1,
        torch.arange(len(inliers)).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img0),
        K.tensor_to_image(img1),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': True})

    plt.savefig(args.output_dir)


if __name__ == '__main__':
    main()

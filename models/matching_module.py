import gc

import pytorch_lightning as pl
import torch

from models.features import get_feature_extractor
from models.features.utils import prepare_features_output
from models.gt_matches_generation import generate_gt_matches
from models.laf_converter import get_laf_to_sideinfo_converter
from models.superglue.superglue import SuperGlue
from utils.augmentations import get_augmentation_transform
from utils.losses import criterion
from utils.metrics import AccuracyUsingEpipolarDist, CameraPoseAUC
from utils.misc import arange_like


class MatchingTrainingModule(pl.LightningModule):
    def __init__(self, train_config, features_config, superglue_config):
        super(MatchingTrainingModule, self).__init__()
        self.config = train_config
        self.features_config = features_config
        self.superglue_config = superglue_config

        if not self.config.get('use_cached_features', False):
            self.local_features_extractor = get_feature_extractor(self.features_config['name'])(
                **self.features_config['parameters']
            )

            self.finetune_features_extractor = self.features_config.get('finetune', False)
            for p in self.local_features_extractor.parameters():
                p.requires_grad = self.finetune_features_extractor

        # The dimensionality of descriptor depends on the local feature extractor, so this parameter is
        # configures in features config file
        descriptor_dim = self.features_config['descriptor_dim']
        self.superglue_config['descriptor_dim'] = descriptor_dim
        self.superglue_config['positional_encoding']['output_size'] = descriptor_dim
        self.superglue_config['attention_gnn']['embed_dim'] = descriptor_dim

        self.laf_converter = get_laf_to_sideinfo_converter(self.superglue_config['laf_to_sideinfo_method'])
        # set side_info dimension based on provided laf_converter
        self.superglue_config['positional_encoding']['side_info_size'] = \
            self.laf_converter.side_info_dim + 1  # plus 1 for responses
        self.superglue = SuperGlue(self.superglue_config)

        # augmentation
        self.augmentations = get_augmentation_transform(self.config['augmentations'])

        # metrics
        if self.config.get('evaluation', True):
            self.epipolar_dist_metric = AccuracyUsingEpipolarDist(self.config['epipolar_dist_threshold'])
            self.camera_pose_auc_metric = CameraPoseAUC(self.config['camera_auc_thresholds'],
                                                        self.config['camera_auc_ransac_inliers_threshold'])

    def augment(self, batch):
        """Augment images and update intrinsic matrices with corresponding geometric transformations"""
        image0_aug = self.augmentations(batch['image0'])
        transform0 = self.augmentations.transform_matrix
        image1_aug = self.augmentations(batch['image1'])
        transform1 = self.augmentations.transform_matrix

        batch['image0'] = image0_aug
        batch['image1'] = image1_aug

        if 'K0' in batch['transformation'] and transform0 is not None:
            batch['transformation']['K0'] = torch.matmul(transform0, batch['transformation']['K0'])
        if 'K1' in batch['transformation'] and transform1 is not None:
            batch['transformation']['K1'] = torch.matmul(transform1, batch['transformation']['K1'])
        return batch

    def training_step(self, batch, batch_idx):
        if not self.config.get('use_cached_features', False):  # if online feature detection is used
            with torch.no_grad():
                batch = self.augment(batch)

            # set eval mode of feature extractor if fine-tuning is off
            if not self.finetune_features_extractor:
                self.local_features_extractor.eval()

            lafs0, responses0, desc0 = self.local_features_extractor(batch['image0'])
            lafs1, responses1, desc1 = self.local_features_extractor(batch['image1'])
        else:
            lafs0, responses0, desc0 = batch['lafs0'], batch['scores0'], batch['descriptors0']
            lafs1, responses1, desc1 = batch['lafs1'], batch['scores1'], batch['descriptors1']

        log_transform_response = self.superglue_config.get('log_transform_response', False)
        data, y_true = generate_gt_matches(
            batch,
            prepare_features_output(lafs0, responses0, desc0, self.laf_converter, log_response=log_transform_response),
            prepare_features_output(lafs1, responses1, desc1, self.laf_converter, log_response=log_transform_response),
            self.config['gt_positive_threshold'],
            self.config['gt_negative_threshold']
        )

        # skip step if no keypoints are detected on at least one of the images
        if data is None:
            return None

        y_pred = self.superglue(data)

        loss = criterion(y_true, y_pred, margin=self.config['margin'])
        self.log('Train NLL loss', loss['loss'].detach(), prog_bar=True, sync_dist=True, on_epoch=False)
        self.log('Train Metric loss', loss['metric_loss'].detach(), prog_bar=True, sync_dist=True, on_epoch=False)

        return self.config['nll_weight'] * loss['loss'] + self.config['metric_weight'] * loss['metric_loss']

    def validation_step(self, batch, batch_idx):
        transformation = {k: v[0] for k, v in batch['transformation'].items()}

        with torch.no_grad():
            pred = self.forward(batch)
        pred = {k: v[0] for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        matched_mask = matches > -1
        matched_kpts0 = kpts0[matched_mask]
        matched_kpts1 = kpts1[matches[matched_mask]]

        self.epipolar_dist_metric(matched_kpts0, matched_kpts1, transformation, len(kpts0))
        self.camera_pose_auc_metric(matched_kpts0, matched_kpts1, transformation)

        return {'matched_kpts0': matched_kpts0, 'matched_kpts1': matched_kpts1,
                'transformation': transformation, 'num_detected_kpts': len(kpts0)}

    def on_validation_epoch_end(self):
        self.log_dict(self.epipolar_dist_metric.compute(), sync_dist=True)
        self.log_dict(self.camera_pose_auc_metric.compute(), sync_dist=True)
        self.epipolar_dist_metric.reset()
        self.camera_pose_auc_metric.reset()
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.superglue.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=self.config['scheduler_gamma']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def forward(self, data):
        # run feature extractor on both images
        pred = {}
        if not self.config.get('use_cached_features', False):
            lafs0, responses0, desc0 = self.local_features_extractor(data['image0'])
            lafs1, responses1, desc1 = self.local_features_extractor(data['image1'])
        else:
            lafs0, responses0, desc0 = data['lafs0'], data['scores0'], data['descriptors0']
            lafs1, responses1, desc1 = data['lafs1'], data['scores1'], data['descriptors1']

        log_transform_response = self.superglue_config.get('log_transform_response', False)
        features0 = prepare_features_output(lafs0, responses0, desc0, self.laf_converter,
                                            log_response=log_transform_response)
        features1 = prepare_features_output(lafs1, responses1, desc1, self.laf_converter,
                                            log_response=log_transform_response)
        pred = {**pred, **{k + '0': v for k, v in features0.items()}}
        pred = {**pred, **{k + '1': v for k, v in features1.items()}}
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)) and isinstance(data[k][0], torch.Tensor):
                data[k] = torch.stack(data[k])

        # predict matching scores
        scores = self.superglue(data)['scores']

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))

        return {
            **pred,
            'matches0': indices0,  # use -1 for invalid match
            'matching_scores0': mscores0,
        }

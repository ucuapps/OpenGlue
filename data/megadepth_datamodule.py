from typing import Optional

import pytorch_lightning as pl
import torch
from functools import partial
from torch.utils.data import DataLoader

from data.megadepth_dataset import MegaDepthPairsDataset, MegaDepthPairsDatasetFeatures
from data.megadepth_balanced_sampler import MegaDepthBalancedSampler


class BaseMegaDepthPairsDataModule(pl.LightningDataModule):
    def __init__(self, root_path, train_list_path, val_list_path, test_list_path,
                 batch_size, num_workers, val_max_pairs_per_scene, balanced_train=False,
                 train_pairs_overlap=None):
        super(BaseMegaDepthPairsDataModule, self).__init__()
        self.root_path = root_path
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.test_list_path = test_list_path

        self.balanced_train = balanced_train
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.val_max_pairs_per_scene = val_max_pairs_per_scene
        self.train_pairs_overlap = train_pairs_overlap

        self.train_batch_collate_fn = None
        self.val_batch_collate_fn = None

    @staticmethod
    def read_scenes_list(path):
        with open(path) as f:
            scenes_list = f.readlines()
        return [s.rstrip() for s in scenes_list]

    def train_dataloader(self):
        sampler = MegaDepthBalancedSampler(self.train_ds) if self.balanced_train else None
        return DataLoader(
            self.train_ds,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_batch_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=self.val_batch_collate_fn
        )


class MegaDepthPairsDataModule(BaseMegaDepthPairsDataModule):
    def __init__(self, root_path, train_list_path, val_list_path, test_list_path,
                 batch_size, num_workers, target_size, val_max_pairs_per_scene,
                 balanced_train=False, train_pairs_overlap=None):
        super(MegaDepthPairsDataModule, self).__init__(
            root_path, train_list_path, val_list_path, test_list_path,
            batch_size, num_workers, val_max_pairs_per_scene, balanced_train,
            train_pairs_overlap
        )
        self.target_size = target_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = MegaDepthPairsDataset(
            root_path=self.root_path,
            scenes_list=self.read_scenes_list(self.train_list_path),
            target_size=self.target_size,
            random_crop=True,
            overlap=self.train_pairs_overlap
        )
        self.val_ds = MegaDepthPairsDataset(
            root_path=self.root_path,
            scenes_list=self.read_scenes_list(self.val_list_path),
            target_size=self.target_size,
            random_crop=False,
            max_pairs_per_scene=self.val_max_pairs_per_scene
        )


class MegaDepthPairsDataModuleFeatures(BaseMegaDepthPairsDataModule):
    def __init__(self, root_path, train_list_path, val_list_path, test_list_path,
                 batch_size, num_workers, target_size, features_dir, num_keypoints, val_max_pairs_per_scene,
                 balanced_train=False, train_pairs_overlap=None):
        super(MegaDepthPairsDataModuleFeatures, self).__init__(
            root_path, train_list_path, val_list_path, test_list_path,
            batch_size, num_workers, val_max_pairs_per_scene, balanced_train,
            train_pairs_overlap
        )
        self.features_dir = features_dir
        self.target_size = target_size
        self.num_keypoints = num_keypoints

        self.train_batch_collate_fn = partial(self.stack_keypoints_batch, target_num_keypoints=num_keypoints,
                                              random=True)
        self.val_batch_collate_fn = partial(self.stack_keypoints_batch, target_num_keypoints=num_keypoints,
                                            random=False)

    @staticmethod
    def stack_keypoints_batch(batch, target_num_keypoints, random=False):
        """
        Stacks keypoints, descriptors, scores and transformations into batch such that each element contains
        equal number of keypoints (target_num_keypoints).
        If the present number of keypoints is bigger than target_num_keypoints, keypoints are select either randomly
        or by top confidence depending on the `random` flag
        if the present number of keypoints is smaller than target_num_keypoints,
        virtual keypoints are added with depth=0, thus they will be ignored during training
        """
        batch_size = len(batch)
        descriptor_size = batch[0]['descriptors0'].size(1)

        result = {
            'lafs0': torch.zeros(batch_size, target_num_keypoints, 2, 3),
            'scores0': torch.zeros(batch_size, target_num_keypoints),
            'descriptors0': torch.zeros(batch_size, target_num_keypoints, descriptor_size),
            'lafs1': torch.zeros(batch_size, target_num_keypoints, 2, 3),
            'scores1': torch.zeros(batch_size, target_num_keypoints),
            'descriptors1': torch.zeros(batch_size, target_num_keypoints, descriptor_size),
            'image0_size': batch[0]['image0_size'],
            'image1_size': batch[0]['image1_size'],
        }
        transformation = {
            'type': ['3d_reprojection'],
            'K0': torch.stack([x['transformation']['K0'] for x in batch]),
            'K1': torch.stack([x['transformation']['K1'] for x in batch]),
            'R': torch.stack([x['transformation']['R'] for x in batch]),
            'T': torch.stack([x['transformation']['T'] for x in batch]),
            'depth0': torch.zeros(batch_size, target_num_keypoints),
            'depth1': torch.zeros(batch_size, target_num_keypoints)
        }

        for i in range(batch_size):
            for image_id in (0, 1):
                num_kpts = batch[i][f'lafs{image_id}'].size(0)

                if num_kpts > target_num_keypoints:  # select subset of keypoint
                    if random:  # select randomly
                        kpts_select_idx = torch.randperm(num_kpts)[:target_num_keypoints]
                    else:  # select based on top confidence
                        kpts_select_idx = torch.topk(batch[i][f'scores{image_id}'], target_num_keypoints, dim=0).indices

                    result[f'lafs{image_id}'][i] = batch[i][f'lafs{image_id}'][kpts_select_idx]
                    result[f'scores{image_id}'][i] = batch[i][f'scores{image_id}'][kpts_select_idx]
                    result[f'descriptors{image_id}'][i] = batch[i][f'descriptors{image_id}'][kpts_select_idx]

                    transformation[f'depth{image_id}'][i] = batch[i]['transformation'][f'depth{image_id}'][
                        result[f'lafs{image_id}'][i][:, 1, 2].type(torch.int64),
                        result[f'lafs{image_id}'][i][:, 0, 2].type(torch.int64),
                    ]

                else:  # select all keypoint and treat other kpts as virtual which are ignored while training
                    result[f'lafs{image_id}'][i, :num_kpts] = batch[i][f'lafs{image_id}']
                    result[f'scores{image_id}'][i, :num_kpts] = batch[i][f'scores{image_id}']
                    result[f'descriptors{image_id}'][i, :num_kpts] = batch[i][f'descriptors{image_id}']

                    transformation[f'depth{image_id}'][i, :num_kpts] = batch[i]['transformation'][f'depth{image_id}'][
                        batch[i][f'lafs{image_id}'][:, 1, 2].type(torch.int64),
                        batch[i][f'lafs{image_id}'][:, 0, 2].type(torch.int64),
                    ]

        result['transformation'] = transformation

        return result

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = MegaDepthPairsDatasetFeatures(
            root_path=self.root_path,
            features_dir=self.features_dir,
            scenes_list=self.read_scenes_list(self.train_list_path),
            target_size=self.target_size,
            random_crop=True,
            overlap=self.train_pairs_overlap
        )
        self.val_ds = MegaDepthPairsDatasetFeatures(
            root_path=self.root_path,
            features_dir=self.features_dir,
            scenes_list=self.read_scenes_list(self.val_list_path),
            target_size=self.target_size,
            random_crop=False,
            max_pairs_per_scene=self.val_max_pairs_per_scene
        )


if __name__ == '__main__':
    dm = MegaDepthPairsDataModuleFeatures(
        root_path='/datasets/extra_space2/ostap/MegaDepth',
        train_list_path='/home/ostap/projects/superglue-lightning/assets/megadepth_train_3.0.txt',
        val_list_path='/home/ostap/projects/superglue-lightning/assets/megadepth_valid_3.0.txt',
        test_list_path='/home/ostap/projects/superglue-lightning/assets/megadepth_valid_3.0.txt',
        batch_size=12, num_workers=3, num_keypoints=1024,
        features_dir='SuperPointNet_960_720',
        target_size=[960, 720],
        val_max_pairs_per_scene=50,
        train_pairs_overlap=[0.15, 0.7]
    )

    dm.setup()
    val_dl = dm.val_dataloader()
    batch = next(iter(val_dl))
    print(batch)

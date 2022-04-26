import glob
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import cv2
import deepdish as dd
import numpy as np
import torch


def array_to_tensor(img_array):
    return torch.FloatTensor(img_array / 255.).unsqueeze(0)


class MegaDepthWarpingDataset(torch.utils.data.Dataset):
    """
    MegaDepth dataset that creates images pair by warping single image.
    """

    def __init__(self, root_path, scenes_list, target_size):
        self.root_path = Path(root_path)
        self.images_list = [  # iter through all scenes and concatenate the results into one list
            *chain(*[glob.glob(
                str(self.root_path / scene / 'dense*' / 'imgs' / '*')
            ) for scene in scenes_list])
        ]
        self.target_size = tuple(target_size)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.target_size != -1:
            image = cv2.resize(image, self.target_size)

        # warp image with random perspective transformation
        height, width = image.shape
        corners = np.array([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
        warp_offset = np.random.randint(-300, 300, size=(4, 2)).astype(np.float32)

        H = cv2.getPerspectiveTransform(corners, corners + warp_offset)
        warped = cv2.warpPerspective(src=image, M=H, dsize=(width, height))
        transformation = {
            'type': 'perspective',
            'H': torch.FloatTensor(H)
        }

        return {'image0': array_to_tensor(image), 'image1': array_to_tensor(warped), 'transformation': transformation}


class BaseMegaDepthPairsDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, scenes_list, max_pairs_per_scene=None, overlap=None):
        self.root_path = Path(root_path)

        pairs_metadata_files = {scene: self.root_path / 'pairs' / scene / 'sparse-txt' / 'pairs.txt' for scene
                                in scenes_list}
        self.image_pairs = OrderedDict()
        for scene, pairs_path in pairs_metadata_files.items():
            try:
                with open(pairs_path) as f:
                    pairs_metadata = f.readlines()
                    pairs_metadata = list(map(lambda x: x.rstrip(), pairs_metadata))
                    if overlap is not None:  # keep pairs with given overlap
                        pairs_metadata = self.filter_pairs_by_overlap(pairs_metadata, overlap)
            except FileNotFoundError:
                pairs_metadata = []
            self.image_pairs[scene] = pairs_metadata
        self.scene_pairs_numbers = OrderedDict([(k, len(v)) for k, v in self.image_pairs.items()])

        if max_pairs_per_scene is not None:  # validation
            self.scene_pairs_numbers = {k: min(v, max_pairs_per_scene) for k, v in self.scene_pairs_numbers.items()}

    def __len__(self):
        return sum(self.scene_pairs_numbers.values())

    def __getitem__(self, idx):
        for s, pairs_num in self.scene_pairs_numbers.items():
            if idx < pairs_num:
                scene, scene_idx = s, idx
                break
            else:
                idx -= pairs_num
        metadata = self.image_pairs[scene][scene_idx]
        return self.parse_pairs_line(metadata), scene, scene_idx

    @staticmethod
    def parse_pairs_line(line):
        img0_name, img1_name, _, _, *camera_params, overlap = line.split(' ')
        camera_params = list(map(lambda x: float(x), camera_params))
        K0, K1, RT = camera_params[:9], camera_params[9:18], camera_params[18:]
        K0 = np.array(K0).astype(np.float32).reshape(3, 3)
        K1 = np.array(K1).astype(np.float32).reshape(3, 3)
        RT = np.array(RT).astype(np.float32).reshape(4, 4)
        R, T = RT[:3, :3], RT[:3, 3]
        return img0_name, img1_name, K0, K1, R, T, float(overlap)

    @staticmethod
    def filter_pairs_by_overlap(pairs_metadata, overlap_range):
        result = []
        min_overlap, max_overlap = overlap_range
        for line in pairs_metadata:
            overlap = float(line.split(' ')[-1])
            if min_overlap <= overlap <= max_overlap:
                result.append(line)
        return result


class MegaDepthPairsDataset(BaseMegaDepthPairsDataset):
    def __init__(self, root_path, scenes_list, target_size=None, random_crop=False, max_pairs_per_scene=None,
                 overlap=None):
        super(MegaDepthPairsDataset, self).__init__(root_path, scenes_list, max_pairs_per_scene, overlap)
        self.target_size = tuple(target_size) if target_size is not None else None
        self.random_crop = random_crop

    def __getitem__(self, idx):
        (img0_name, img1_name, K0, K1, R, T, overlap), \
        scene, scene_idx = super(MegaDepthPairsDataset, self).__getitem__(idx)

        # read and transform images
        images = []
        for img_name, K in ((img0_name, K0), (img1_name, K1)):
            image = cv2.imread(str(self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/imgs' / img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            depth = dd.io.load(str(
                self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/depths' / (img_name[:-3] + 'h5')))[
                'depth']

            if self.target_size is not None:
                size = image.shape[:2][::-1]
                current_ratio = size[0] / size[1]
                target_ratio = self.target_size[0] / self.target_size[1]

                if current_ratio > target_ratio:
                    resize_height = self.target_size[1]
                    resize_width = int(current_ratio * resize_height)

                    image = cv2.resize(image, (resize_width, resize_height))
                    depth = cv2.resize(depth, (resize_width, resize_height), cv2.INTER_NEAREST)
                    # crop width
                    # max fixes case when resize_width == self.target_size[0]
                    if self.random_crop:
                        start_width = np.random.randint(0, max(resize_width - self.target_size[0], 1))
                    else:
                        start_width = (resize_width - self.target_size[0]) // 2
                    end_width = start_width + self.target_size[0]

                    image = image[:, start_width:end_width]
                    depth = depth[:, start_width:end_width]
                    # update K
                    scales = np.diag([resize_width / size[0], resize_height / size[1], 1.0]).astype(np.float32)
                    K = np.dot(scales, K)
                    K[0, 2] -= start_width
                else:
                    resize_width = self.target_size[0]
                    resize_height = int(resize_width / current_ratio)

                    image = cv2.resize(image, (resize_width, resize_height))
                    depth = cv2.resize(depth, (resize_width, resize_height), cv2.INTER_NEAREST)
                    # crop height
                    if self.random_crop:
                        start_height = np.random.randint(0, max(resize_height - self.target_size[1], 1))
                    else:
                        start_height = (resize_height - self.target_size[1]) // 2
                    end_height = start_height + self.target_size[1]

                    image = image[start_height:end_height, :]
                    depth = depth[start_height:end_height, :]
                    # update K
                    scales = np.diag([resize_width / size[0], resize_height / size[1], 1.0]).astype(np.float32)
                    K = np.dot(scales, K)
                    K[1, 2] -= start_height

            images.append((image, depth, K))

        (image0, depth0, K0), (image1, depth1, K1) = images

        transformation = {
            'type': '3d_reprojection',
            'K0': torch.from_numpy(K0),
            'K1': torch.from_numpy(K1),
            'R': torch.from_numpy(R),
            'T': torch.from_numpy(T),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
        }

        return {'image0': array_to_tensor(image0), 'image1': array_to_tensor(image1), 'transformation': transformation}


class MegaDepthPairsDatasetFeatures(BaseMegaDepthPairsDataset):
    def __init__(self, root_path, features_dir, scenes_list, target_size=None, random_crop=False,
                 max_pairs_per_scene=None, overlap=None):
        super(MegaDepthPairsDatasetFeatures, self).__init__(root_path, scenes_list, max_pairs_per_scene, overlap)
        self.features_base_dir = self.root_path / features_dir
        self.target_size = tuple(target_size) if target_size is not None else None
        self.random_crop = random_crop

    def __getitem__(self, idx):
        (img0_name, img1_name, K0, K1, R, T, overlap), \
        scene, scene_idx = super(MegaDepthPairsDatasetFeatures, self).__getitem__(idx)

        images_info = []
        for img_name, K in ((img0_name, K0), (img1_name, K1)):
            base_name = img_name[:-4]
            image = cv2.imread(str(self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/imgs' / img_name))
            orig_size = image.shape[:2][::-1]  # original image size

            features_dir = self.features_base_dir / scene

            lafs = dd.io.load(features_dir / f'{base_name}_lafs.h5')
            scores = dd.io.load(features_dir / f'{base_name}_scores.h5')
            descriptors = dd.io.load(features_dir / f'{base_name}_descriptors.h5', )
            # size after resizing one of sides to target size
            image_size = dd.io.load(features_dir / f'{base_name}_size.h5')

            depth = dd.io.load(
                self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/depths' / f'{base_name}.h5')['depth']
            depth = cv2.resize(depth, tuple(image_size), interpolation=cv2.INTER_NEAREST)

            scales = np.diag([image_size[0] / orig_size[0], image_size[1] / orig_size[1], 1.0]).astype(np.float32)
            K = np.dot(scales, K)

            # perform random crop
            if self.target_size[0] < image_size[0]:  # crop by width
                if self.random_crop:
                    start_width = np.random.randint(0, image_size[0] - self.target_size[0])
                else:
                    start_width = (image_size[0] - self.target_size[0]) // 2
                end_width = start_width + self.target_size[0]

                depth = depth[:, start_width:end_width]
                kpts_crop_mask = (lafs[:, 0, 2] >= start_width) & (lafs[:, 0, 2] < end_width)
                K[0, 2] -= start_width

                lafs = lafs[kpts_crop_mask]
                lafs[:, 0, 2] -= start_width
                scores = scores[kpts_crop_mask]
                descriptors = descriptors[kpts_crop_mask]
            elif self.target_size[1] < image_size[1]:  # crop by height
                if self.random_crop:
                    start_height = np.random.randint(0, image_size[1] - self.target_size[1])
                else:
                    start_height = (image_size[1] - self.target_size[1]) // 2
                end_height = start_height + self.target_size[1]

                depth = depth[start_height:end_height, :]
                kpts_crop_mask = (lafs[:, 1, 2] >= start_height) & (lafs[:, 1, 2] < end_height)
                K[1, 2] -= start_height

                lafs = lafs[kpts_crop_mask]
                lafs[:, 1, 2] -= start_height
                scores = scores[kpts_crop_mask]
                descriptors = descriptors[kpts_crop_mask]

            images_info.append((lafs, scores, descriptors, depth, K))

        (lafs0, scores0, descriptors0, depth0, K0), (lafs1, scores1, descriptors1, depth1, K1) = images_info
        transformation = {
            'type': '3d_reprojection',
            'K0': torch.from_numpy(K0),
            'K1': torch.from_numpy(K1),
            'R': torch.from_numpy(R),
            'T': torch.from_numpy(T),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
        }
        return {
            'lafs0': torch.from_numpy(lafs0),
            'scores0': torch.from_numpy(scores0),
            'descriptors0': torch.from_numpy(descriptors0),
            'lafs1': torch.from_numpy(lafs1),
            'scores1': torch.from_numpy(scores1),
            'descriptors1': torch.from_numpy(descriptors1),
            'transformation': transformation,
            'image0_size': self.target_size,
            'image1_size': self.target_size
        }


if __name__ == '__main__':
    with open('/home/ostap/projects/superglue-lightning/assets/megadepth_valid_2.0.txt') as f:
        scenes_list = f.readlines()

    ds = MegaDepthPairsDatasetFeatures(
        root_path='/datasets/extra_space2/ostap/MegaDepth',
        features_dir='SuperPointNet_960_720',
        target_size=[960, 720],
        random_crop=True,
        scenes_list=[s.rstrip() for s in scenes_list],
        overlap=[0.15, 0.7]
    )
    print(ds[100]['transformation']['depth0'].size())

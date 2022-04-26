import glob
import pathlib
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch


class OxfordParis1MDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: pathlib.Path, resize_shape: Tuple[int, int], offset: int):
        self.root_path = root_path
        self.resize_shape = resize_shape
        self.offset = offset
        self.images_list = glob.glob(str(root_path / '*' / '*.jpg'))

        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.25),
            A.ColorJitter(p=0.15),
            A.GaussNoise(p=0.25)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx: int):
        image_path = self.images_list[idx]
        image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, self.resize_shape, interpolation=cv2.INTER_AREA)

        width, height = self.resize_shape
        width_target, height_target = width - 2 * self.offset, height - 2 * self.offset
        corners_dst = np.array([
            [self.offset, self.offset],
            [self.offset, height - self.offset - 1],
            [width - self.offset - 1, self.offset],
            [width - self.offset - 1, height - self.offset - 1]
        ], dtype=np.float32)

        warp_offset = np.random.randint(-self.offset, self.offset, size=(4, 2)).astype(np.float32)
        corners_src = corners_dst + warp_offset
        H_warp = cv2.getPerspectiveTransform(corners_src, corners_dst)  # used for warping image

        corners_dst = np.array(
            [[0, 0], [0, height_target - 1], [width_target - 1, 0], [width_target - 1, height_target - 1]],
            dtype=np.float32)
        corners_src = corners_dst + warp_offset
        H_true = cv2.getPerspectiveTransform(corners_src, corners_dst)  # used for relating crops of image

        image2 = cv2.warpPerspective(src=image1, M=H_warp, dsize=(width, height))

        image1 = image1[self.offset:height - self.offset, self.offset:width - self.offset]
        image2 = image2[self.offset:height - self.offset, self.offset:width - self.offset]

        transformation = {
            'type': 'perspective',
            'H': torch.FloatTensor(H_true)
        }
        image1, image2 = map(
            lambda x: (torch.FloatTensor(
                cv2.cvtColor(self.color_aug(image=x)['image'], cv2.COLOR_RGB2GRAY)
            ) / 255.).unsqueeze(0),
            (image1, image2)
        )
        return {'image0': image1, 'image1': image2, 'transformation': transformation}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = OxfordParis1MDataset(pathlib.Path('/datasets/extra_space2/ostap/temp_dataset'), resize_shape=(960, 720),
                              offset=100)
    x = ds[139]
    image1, image2, H = x['image0'], x['image1'], x['transformation']['H']
    print(image1.shape, image2.shape)

    image1, image2 = map(
        lambda x: (x * 255).type(torch.uint8).numpy()[0],
        (image1, image2)
    )
    H = H.numpy()
    pt1 = np.array([300., 400., 1.])
    pt2 = H @ pt1
    pt2 /= pt2[-1]
    print(pt2)
    cv2.circle(image1, (int(pt1[0]), int(pt1[1])), radius=3, color=(255, 0, 0), thickness=3)
    cv2.circle(image2, (int(pt2[0]), int(pt2[1])), radius=3, color=(255, 0, 0), thickness=3)

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(image1)
    axes[1].imshow(image2)
    plt.show()

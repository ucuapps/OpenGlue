import argparse
import glob
import logging
import math
import pathlib
from typing import Tuple, List, Union, Optional

import cv2
import deepdish as dd
import numpy as np
import os
import torch
import yaml
from torch import multiprocessing

from models.features import get_feature_extractor

logging.basicConfig(format='[%(asctime)s] %(name)s | %(levelname)s: %(message)s', datefmt='%Y/%m/%d %I:%M:%S',
                    level=logging.INFO)


def parse_arguments() -> argparse.Namespace:
    """
    Define and parse command line arguments for this module.
    Returns:
        args: Namespace object containing parsed arguments
    """
    parser = argparse.ArgumentParser('Local Features Extraction')
    parser.add_argument(
        '--device',
        help='Device type, where features are extracted',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
    )
    parser.add_argument(
        '--num_workers',
        help='Number of workers for parallel processing',
        type=int,
        default=1
    )
    parser.add_argument(
        '--target_size',
        help='Target size of the image (WIDTH, HEIGHT). '
             'At least one side of resulting image will correspond with `target_size` '
             'dimensions, such that the original aspect ration is preserved',
        type=int,
        default=None,
        nargs=2
    )
    parser.add_argument(
        '--data_path',
        help='Path to directory with scenes images',
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        '--output_path',
        help='Path to directory where extracted features are stored',
        type=pathlib.Path,
        required=True
    )
    parser.add_argument(
        '--extractor_config_path',
        help='Path to the file containing config for feature extractor in .yaml format',
        type=pathlib.Path,
        default='config/features/sift_opencv.yaml'
    )
    parser.add_argument(
        '--recompute',
        help='Flag indicating whether to recompute features if it is already present in output directory',
        action='store_true'
    )
    parser.add_argument(
        '--image_format',
        help='Formats of images searched inside `data_path` to compute features for',
        type=str,
        default=['jpg', 'JPEG', 'JPG', 'png'],
        nargs='+'
    )
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    logger.info(args)

    num_workers = args.num_workers
    if args.device == 'cuda' and torch.cuda.device_count() < num_workers:
        logger.warning(f'Number of workers selected is bigger than number of available cuda devices. '
                       f'Setting num_workers to {torch.cuda.device_count()}.')
        num_workers = torch.cuda.device_count()

    # read feature extractor config
    with open(args.extractor_config_path) as f:
        feature_extractor_config = yaml.full_load(f)

    # make output directory
    output_path = args.output_path / get_output_directory_name(feature_extractor_config, args)
    logger.info(f'Creating output directory {output_path} (if not exists).')
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        yaml.dump(feature_extractor_config, f)

    images_list = get_images_list(args.data_path, args.image_format)
    logger.info(f'Total number of images found to process: {len(images_list)}')
    # split into chunks of (almost) equal size
    chunk_size = math.ceil(len(images_list) / num_workers)
    images_list = [images_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    logger.info(f'Starting {num_workers} processes for features extraction.')
    multiprocessing.start_processes(
        process_chunk,
        args=(images_list, feature_extractor_config, output_path, args),
        nprocs=num_workers,
        join=True
    )


def process_chunk(process_id: int, images_list: List[Tuple[str, Union[str, None]]], feature_extractor_config: dict,
                  output_path: Union[str, pathlib.Path], args: argparse.Namespace):
    """Function to execute on each worker"""
    if args.device == 'cuda':
        device = f'cuda:{process_id}'
    else:
        device = 'cpu'

    cv2.setNumThreads(1)
    logger = logging.getLogger(__name__)

    features_name = feature_extractor_config['name']
    feature_extractor = get_feature_extractor(features_name)(**feature_extractor_config['parameters'])
    feature_extractor.eval().to(device)

    with torch.inference_mode():
        images_list = images_list[process_id]
        for i, (image_path, scene) in enumerate(images_list, start=1):
            output_path_scene = output_path
            if scene is not None:
                output_path_scene = output_path_scene / scene

            os.makedirs(output_path_scene, exist_ok=True)

            base_name = image_path.rpartition(os.path.sep)[2].rpartition('.')[0]
            # skip image if output already exists and recompute=False
            if not args.recompute and check_if_features_exist(output_path_scene, base_name):
                continue

            image = read_image(image_path, args.target_size)
            image = (torch.FloatTensor(image) / 255.).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            _, _, resize_height, resize_width = image.size()
            lafs, scores, descriptors = map(lambda x: x[0].cpu().numpy(), feature_extractor(image))

            # save results
            save_outputs(
                output_path_scene,
                base_name,
                (lafs, scores, descriptors, np.array([resize_width, resize_height]))
            )

            if i % 100 == 0:
                logger.info(f'PID #{process_id}: Processed {i}/{len(images_list)} images.')


def get_output_directory_name(feature_extractor_config: dict, args: argparse.Namespace) -> str:
    """
    Build output directory name based on parameters
    Args:
        feature_extractor_config: parameters of feature extractor
        args: command line arguments

    Returns:
        Name of directory where features are stored
    """
    name = feature_extractor_config['name']
    if args.target_size is not None:
        name += f'_{args.target_size[0]}_{args.target_size[1]}'
    return name


def get_images_list(input_data_path: pathlib.Path, image_formats: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Get list of images that wil be processed.
    Args:
        input_data_path: input path to the location with images
        image_formats: file formats of images to look for
    Returns:
        images_path_list: list where each image is represented as tuple with path to image
        and scene name (or None if no scenes are available)
    """
    # process each scene
    images_path_list = []
    scenes = os.listdir(input_data_path)
    for scene in scenes:
        images_path = input_data_path / scene / 'dense0' / 'imgs'
        scene_list = []
        for image_format in image_formats:
            scene_list.extend(glob.glob(str(images_path / f'*.{image_format}')))
        images_path_list.extend(((path, scene) for path in scene_list))
    return images_path_list


def check_if_features_exist(output_path_scene: pathlib.Path, image_base_name: str) -> bool:
    """
    Check if feature outputs already exist in the output location
    Args:
        output_path_scene: location of output for particular scene
        image_base_name: image name without format

    Returns:
        exists_flag: flag indicating whether the outputs for particular base_name exists
    """
    return os.path.exists(output_path_scene / f'{image_base_name}_lafs.h5') and \
           os.path.exists(output_path_scene / f'{image_base_name}_scores.h5') and \
           os.path.exists(output_path_scene / f'{image_base_name}_descriptors.h5') and \
           os.path.exists(output_path_scene / f'{image_base_name}_size.h5')


def read_image(image_path: Union[str, pathlib.Path], target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Read image, convert to gray and resize to target size
    Args:
        image_path: path to image file
        target_size: size of returned image will correspond with target size in at least one dimension, such that
        its aspect ratio is preserved

    Returns:
        image: array (H, W) representing image, where H=target_size[1] or W=target_size[0]
    """
    # read image and convert to gray
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = image.shape[:2][::-1]

    if target_size is not None:
        # resize image to target size
        target_ratio = target_size[0] / target_size[1]  # 960 / 720 = 1.333
        current_ratio = size[0] / size[1]
        if current_ratio > target_ratio:
            resize_height = target_size[1]
            resize_width = int(current_ratio * resize_height)
            image = cv2.resize(image, (resize_width, resize_height))
        else:
            resize_width = target_size[0]
            resize_height = int(resize_width / current_ratio)
            image = cv2.resize(image, (resize_width, resize_height))
    return image


def save_outputs(output_path_scene: pathlib.Path,
                 image_base_name: str, outputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    Safely save outputs. If an error occurs remove all outputs.
    Args:
        output_path_scene: location of output for particular scene
        image_base_name: image name without format
        outputs: outputs to be saved
    """
    lafs, scores, descriptors, size = outputs
    try:
        dd.io.save(output_path_scene / f'{image_base_name}_lafs.h5', lafs)
        dd.io.save(output_path_scene / f'{image_base_name}_scores.h5', scores)
        dd.io.save(output_path_scene / f'{image_base_name}_descriptors.h5', descriptors)
        dd.io.save(output_path_scene / f'{image_base_name}_size.h5', size)
    except (Exception, KeyboardInterrupt):
        os.remove(output_path_scene / f'{image_base_name}_lafs.h5')
        os.remove(output_path_scene / f'{image_base_name}_scores.h5')
        os.remove(output_path_scene / f'{image_base_name}_descriptors.h5')
        os.remove(output_path_scene / f'{image_base_name}_size.h5')
        raise


if __name__ == '__main__':
    main()

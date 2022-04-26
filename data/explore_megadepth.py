import numpy as np
import deepdish as dd
import tqdm
from pathlib import Path
import os
from collections import namedtuple


def parse_intrinsics_line(params_line):
    camera_id, _, width, height, fx, fy, cx, cy = params_line.split(' ')
    width, height, fx, fy, cx, cy = int(width), int(height), float(fx), float(fy), float(cx), float(cy)
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return camera_id, IntrinsicParams(K=K, size=(width, height))


def get_camera2intr(camera_lines):
    camera2intr = {}
    for camera_params_line in tqdm.tqdm(camera_lines):
        if camera_params_line.startswith('#'):
            continue

        camera_id, intrinsics = parse_intrinsics_line(camera_params_line.strip())
        camera2intr[camera_id] = intrinsics
    return camera2intr


def quaternion_rotation_matrix(q0, q1, q2, q3):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def parse_extrinsics_line(params_line):
    image_id, *extrinsics, camera_id, name = params_line.split(' ')
    qw, qx, qy, qz, tx, ty, tz = map(lambda x: float(x), extrinsics)

    R = quaternion_rotation_matrix(qw, qx, qy, qz)
    T = np.array([tx, ty, tz])

    return ExtrinsicParams(R=R, T=T), image_id, camera_id, name


def parse_points(points_line):
    points = []
    points_line = points_line.split(' ')

    for i in range(len(points_line) // 3):
        x, y, point3d_id = points_line[3 * i:3 * (i + 1)]
        x, y = float(x), float(y)
        point = Point2d(x=x, y=y, point3d_id=point3d_id)
        points.append(point)
    return points


def parse_images(images_lines):
    images_lines = images_lines[4:]
    images = []

    for i in tqdm.tqdm(range(len(images_lines) // 2)):
        extr, image_id, camera_id, name = parse_extrinsics_line(images_lines[2 * i].strip())
        points = parse_points(images_lines[2 * i + 1].strip())
        intr = camera2intr[camera_id]
        image = Image(image_id, name, intr, extr, points)
        images.append(image)
    return images


def parse_3d_point(point_line):
    point_id, x, y, z, *_ = point_line.split(' ')
    x, y, z, = float(x), float(y), float(z)
    return Point3d(point_id, x, y, z)

def parse_3d_points(points_lines):
    points = []
    for line in tqdm.tqdm(points_lines):
        if line.startswith('#'): continue
        points.append(parse_3d_point(line))
    return points


def get_points3d_overlap(points1, points2):
    return len(points1 & points2) / min(len(points1), len(points2))


def array2string(arr):
    s = np.array2string(arr.flatten(), max_line_width=10000000)[1:-1].strip()
    # remove unneccesary spaces
    s = ' '.join(filter(lambda x: x != '', s.split(' ')))
    return s


def make_image_pair_record(image1, image2, points3d_overlap):
    # extract intrinsics and extrinsics relative to world coordinates
    path1, path2 = image1.name, image2.name
    K1, R1, T1 = image1.intr.K, image1.extr.R, image1.extr.T
    K2, R2, T2 = image2.intr.K, image2.extr.R, image2.extr.T

    # get relative extrinsics transformation from image 1 to image 2
    R12 = R2 @ R1.T
    T12 = -R12 @ T1 + T2
    RT12 = np.zeros((4, 4), dtype=np.float64)
    RT12[:3, :3] = R12
    RT12[:3, 3] = T12
    RT12[3, 3] = 1

    EXIF1 = EXIF2 = 0

    return f'{path1} {path2} {EXIF1} {EXIF2} {array2string(K1)} {array2string(K2)} {array2string(RT12)} {points3d_overlap}'


def process_scene(images, out_path, targetdir_depth, overlap_interval=[0.1, 0.7]):
    # get only images that have depth map available
    images_valid = []
    for image in images:
        try:
            depth = dd.io.load(targetdir_depth / (image.name[:-4] + '.h5'))['depth']
        except:
            continue
        if np.sum(depth == -1) > 0:
            continue
        images_valid.append(image)
    images = images_valid
    print(len(images))

    image_to_3dpoints = []
    for image in images:
        points_3d = set(map(lambda x: x.point3d_id, filter(lambda x: x.point3d_id != '-1', image.points)))
        image_to_3dpoints.append(points_3d)

    counter = 0
    with open(out_path, 'w') as f:
        for i in tqdm.tqdm(range(len(images))):
            for j in range(i + 1, len(images)):
                image1, image2 = images[i], images[j]
                overlap = get_points3d_overlap(image_to_3dpoints[i], image_to_3dpoints[j])
                if overlap_interval[1] >= overlap >= overlap_interval[0]:
                    record = make_image_pair_record(image1, image2, overlap)
                    counter += 1
                    f.write(record + '\n')
    print(counter)


MEGADEPTH_PATH = 'MegaDepth/'

IntrinsicParams = namedtuple('IntrinsicParams', ['size', 'K'])
ExtrinsicParams = namedtuple('ExtrinsicParams', ['R', 'T'])
Point2d = namedtuple('Point2d', ['x', 'y', 'point3d_id'])
Point3d = namedtuple('Point3d', ['id', 'x', 'y', 'z'])
Image = namedtuple('Image', ['id', 'name', 'intr', 'extr', 'points'])

scenes_list = os.listdir(os.path.join(MEGADEPTH_PATH, 'Undistorted-SfM/'))

for scene in scenes_list[-1:]:
    print(f'Scene {scene}')
    try:
        # skip scene if file pairs.txt already exists
        if 'pairs.txt' in os.listdir(f'{MEGADEPTH_PATH}Undistorted-SfM/{scene}/sparse-txt/'):
            continue
    except FileNotFoundError:
        continue

    targetdir = Path(f'{MEGADEPTH_PATH}Undistorted-SfM/{scene}/sparse-txt/')
    targetdir_depth = Path(f'{MEGADEPTH_PATH}phoenix/S6/zl548/MegaDepth_v1/') / scene / 'dense0' / 'depths'

    with open(targetdir / 'cameras.txt') as f:
        camera_lines = f.readlines()
        camera_lines = list(map(lambda x: x.strip(), camera_lines))

        camera2intr = get_camera2intr(camera_lines)

    with open(targetdir / 'images.txt') as f:
        images_lines = f.readlines()
        images = parse_images(images_lines)

    process_scene(images, targetdir / 'pairs.txt', targetdir_depth, overlap_interval=[0.1, 0.7])

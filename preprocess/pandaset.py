import argparse
import json

import h5py
import gzip
import os
import random
import torch

from pandaset import DataSet, geometry
import numpy as np
import open3d as o3d

from pandaset.geometry import _heading_position_to_mat
from tqdm import tqdm

from utils import to_rotation_matrix


def projection_torch(lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=True):
    camera_heading = camera_pose['heading']
    camera_position = camera_pose['position']
    camera_pose_mat = _heading_position_to_mat(camera_heading, camera_position)

    trans_lidar_to_camera = torch.tensor(camera_pose_mat, dtype=lidar_points.dtype, device=lidar_points.device).inverse()
    points3d_lidar = lidar_points
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                      trans_lidar_to_camera[:3, 3].view(3, 1)

    K = torch.eye(3, dtype=lidar_points.dtype, device=lidar_points.device)
    K[0, 0] = camera_intrinsics.fx
    K[1, 1] = camera_intrinsics.fy
    K[0, 2] = camera_intrinsics.cx
    K[1, 2] = camera_intrinsics.cy

    inliner_indices_arr = torch.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

    return points2d_camera, points3d_camera, inliner_indices_arr

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--voxel_size', default=0.1, help='Voxel Size')
parser.add_argument('--base_folder', default='/data/pandaset',
                    help='Base argoverse folder')
args = parser.parse_args()

device = args.device
data_path = args.base_folder
datasets = DataSet(data_path)

color = 'intensity'  # 'intensity', None

i = 0
for seq_num in tqdm(datasets.sequences(with_semseg=True)):
    print(seq_num)
    i += 1
    for f in os.listdir(os.path.join(args.base_folder, seq_num, 'lidar')):
        if f.endswith('.pkl'):
            with (open(os.path.join(args.base_folder, seq_num, 'lidar', f), 'rb') as f_in,
                  gzip.open(os.path.join(args.base_folder, seq_num, 'lidar', f'{f}.gz'), 'wb') as f_out):
                f_out.writelines(f_in)
            os.remove(os.path.join(args.base_folder, seq_num, 'lidar', f))
    for f in os.listdir(os.path.join(args.base_folder, seq_num, 'annotations', 'semseg')):
        if f.endswith('.pkl'):
            with (open(os.path.join(args.base_folder, seq_num, 'annotations', 'semseg', f), 'rb') as f_in,
                  gzip.open(os.path.join(args.base_folder, seq_num, 'annotations', 'semseg', f'{f}.gz'), 'wb') as f_out):
                f_out.writelines(f_in)
            os.remove(os.path.join(args.base_folder, seq_num, 'annotations', 'semseg', f))
    dataset = DataSet(data_path)
    seq = dataset[seq_num]
    seq.load()
    all_pts = []
    all_colors = []

    for camera_folder in os.listdir(os.path.join(args.base_folder, seq_num, 'camera')):
        poses_torch_folder = os.path.join(args.base_folder, seq_num, 'camera', camera_folder, 'poses_torch')
        if not os.path.exists(poses_torch_folder):
            os.mkdir(poses_torch_folder)
        poses_file = os.path.join(args.base_folder, seq_num, 'camera', camera_folder, 'poses.json')
        poses = json.load(open(poses_file, 'r'))
        camera_frames = os.listdir(os.path.join(args.base_folder, seq_num, 'camera', camera_folder))
        camera_frames = sorted([f for f in camera_frames if f.endswith('.jpg')])
        for frame in camera_frames:
            pose = poses[int(os.path.splitext(os.path.basename(frame))[0])]
            pose_t = torch.tensor([pose['position']['x'], pose['position']['y'], pose['position']['z']])
            pose_r = torch.tensor([pose['heading']['w'], pose['heading']['x'], pose['heading']['y'], pose['heading']['z']])
            pose = to_rotation_matrix(pose_r, pose_t)
            pose_save_path = os.path.join(poses_torch_folder,
                                          os.path.splitext(os.path.basename(frame))[0] + '.npy')
            np.save(pose_save_path, pose.inverse().cpu().numpy())

    continue

    for frame in range(len(seq.lidar.data)):
        pc_np = seq.lidar[frame].values

        static_objects_index = seq.semseg[frame].values[:, 0] < 13
        static_objects_index = static_objects_index | (seq.semseg[frame].values[:, 0] > 33)
        static_objects_index = static_objects_index & (seq.semseg[frame].values[:, 0] != 4)

        pc_np = pc_np[static_objects_index]
        # print(pc_np.shape)

        all_pts.append(pc_np[:, :3].copy())
        all_colors.append(np.stack((pc_np[:, 3], pc_np[:, 3], pc_np[:, 3]), 1).copy())

    pcd = o3d.geometry.PointCloud()
    all_pts = np.concatenate(all_pts, axis=0)
    if color is not None:
        all_colors = np.concatenate(all_colors, axis=0)
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

    downpcd_full = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=0.1)
    cl, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)  # NEW OPEN3D VERSION
    downpcd = downpcd_full.select_by_index(ind)  # NEW OPEN3D VERSION

    all_pts = torch.tensor(downpcd.points, dtype=torch.float).to(device)
    all_colors = torch.tensor(downpcd.colors, dtype=torch.float)[:, 0:1].to(device)

    save_path = os.path.join(data_path, seq_num, 'map.h5')
    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('PC', data=all_pts.cpu().half(), compression='lzf', shuffle=True)
        hf.create_dataset('intensity', data=all_colors.cpu().half(), compression='lzf', shuffle=True)

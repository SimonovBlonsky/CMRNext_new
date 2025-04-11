import argparse
import copy
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import open3d as o3
import pyntcloud
import torch

from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

from utils import to_rotation_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--camera_name', default='image_raw_ring_front_center',
                    help='Camera')
parser.add_argument('--voxel_size', default=0.1, help='Voxel Size')
parser.add_argument('--base_folder', default='/data/argoverse/argoverse-tracking',
                    help='Base argoverse folder')
args = parser.parse_args()

camera_name = args.camera_name

for train_folder in ['train1', 'train2', 'train3', 'train4']:
    dataset_dir = os.path.join(args.base_folcer, train_folder)
    for log_id in os.listdir(dataset_dir):
        print(os.path.join(dataset_dir, log_id))

        save_folder_maps = os.path.join(dataset_dir, log_id, 'point_cloud')
        remove_car = True

        sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)

        city_info_fpath = f"{dataset_dir}/{log_id}/city_info.json"
        city_info = read_json_file(city_info_fpath)
        city_name = city_info["city_name"]
        dataset_map = ArgoverseMap()

        ### CAMERA CALIBRATION
        calib_file = os.path.join(dataset_dir, log_id, 'vehicle_calibration_info.json')
        with open(calib_file, "rb") as f:
            all_calib = json.load(f)
        all_camera_data = all_calib["camera_data_"]
        for camera_data in all_camera_data:
            if camera_name in camera_data["key"]:
                camera_calibration = camera_data["value"]
                break
        else:
            raise ValueError(f"Unknown camera name: {camera_name}")

        T = camera_calibration['vehicle_SE3_camera_']['translation']
        R = camera_calibration['vehicle_SE3_camera_']['rotation']['coefficients']
        v2cam = to_rotation_matrix(torch.tensor(R), torch.tensor(T)).inverse()

        ply_fpaths = sorted(glob.glob(f"{dataset_dir}/{log_id}/lidar/PC_*.ply"))
        img_fpaths = sorted(glob.glob(f"{dataset_dir}/{log_id}/{camera_name[10:]}/{camera_name[10:]}_*.jpg"))
        if len(img_fpaths) == 0:
            img_fpaths = sorted(glob.glob(f"{dataset_dir}/{log_id}/image/{camera_name[10:]}_*.jpg"))
        print(len(img_fpaths))

        all_pts = []
        all_intensities = []
        initial_translation = None
        poses = []

        for i, ply_fpath in enumerate(ply_fpaths):
            # print(ply_fpath)
            lidar_timestamp_ns = ply_fpath.split("/")[-1].split(".")[0].split("_")[-1]

            pose_fpath = f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle_{lidar_timestamp_ns}.json"
            if not Path(pose_fpath).exists():
                print("Not Found!!!, ", pose_fpath)
                continue

            pose_data = read_json_file(pose_fpath)
            rotation = np.array(pose_data["rotation"])
            translation = np.array(pose_data["translation"])
            if i == 0:
                initial_translation = translation
            city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation)
            poses.append(torch.tensor(city_to_egovehicle_se3.transform_matrix).float())

            lidar_pts = load_ply(ply_fpath)

            data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
            x = np.array(data.points.x)[:, np.newaxis]
            y = np.array(data.points.y)[:, np.newaxis]
            z = np.array(data.points.z)[:, np.newaxis]
            lidar_pts = np.concatenate((x, y, z), axis=1)

            lidar_intensity = np.array(data.points.intensity, dtype=np.float32)[:, np.newaxis]
            lidar_intensity = np.concatenate((lidar_intensity, lidar_intensity, lidar_intensity), axis=1)

            lidar_pts = city_to_egovehicle_se3.transform_point_cloud(lidar_pts)

            if remove_car:
                boolean_road_arr = dataset_map.get_raster_layer_points_boolean(lidar_pts, city_name, "driveable_area")
                road_pts = lidar_pts[boolean_road_arr].copy()
                road_intensity = lidar_intensity[boolean_road_arr].copy()

                _, not_ground_logicals = dataset_map.remove_ground_surface(
                    copy.deepcopy(road_pts), city_name, return_logicals=True
                )
                road_pts = road_pts[np.logical_not(not_ground_logicals)]
                road_pts_intensity = road_intensity[np.logical_not(not_ground_logicals)]

                non_road_pts = lidar_pts[np.logical_not(boolean_road_arr)]
                non_road_intensity = lidar_intensity[np.logical_not(boolean_road_arr)]

                all_pts.append(road_pts.copy())
                all_pts.append(non_road_pts.copy())
                all_intensities.append(road_pts_intensity.copy())
                all_intensities.append(non_road_intensity.copy())
            else:
                all_pts.append(lidar_pts.copy())
                all_intensities.append(lidar_intensity.copy())

        all_pts = np.concatenate(all_pts, axis=0)
        all_intensities = np.concatenate(all_intensities, axis=0)
        all_pts -= initial_translation

        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(all_pts)
        pcd.colors = o3.utility.Vector3dVector(all_intensities/255)
        # o3.visualization.draw_geometries([pcd])

        downpcd_full = o3.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=args.voxel_size)
        cl, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)
        downpcd = downpcd_full.select_by_index(ind)

        voxelized = torch.tensor(np.asarray(downpcd.points), dtype=torch.float)
        voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
        voxelized = voxelized.t()
        vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float)[:, 0:1].t()

        single_map_path = os.path.join(dataset_dir, log_id, 'map.h5')
        with h5py.File(single_map_path, 'w') as hf:
            hf.create_dataset('PC', data=voxelized.cpu(), compression='lzf', shuffle=True)
            hf.create_dataset('intensity', data=vox_intensity.cpu(), compression='lzf', shuffle=True)

        initial_translation = torch.tensor(initial_translation).float()
        save_pose_folder = os.path.join(dataset_dir, log_id, camera_name[10:], 'poses_torch')
        if not os.path.exists(save_pose_folder):
            os.mkdir(save_pose_folder)

        print("Saving maps now...")
        for i, img_path in enumerate(img_fpaths):
            img_timestamp = img_path.split("/")[-1].split(".")[0].split("_")[-1]
            pose_fpath = f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle_{img_timestamp}.json"
            if not Path(pose_fpath).exists():
                print("Not Found!!!, ", pose_fpath)
                continue
            pose_data = read_json_file(pose_fpath)
            rotation = np.array(pose_data["rotation"])
            translation = np.array(pose_data["translation"])
            city_to_egovehicle_se3 = SE3(rotation=quat2rotmat(rotation), translation=translation)
            pose = torch.tensor(city_to_egovehicle_se3.transform_matrix).float()

            pose[:3, 3] -= initial_translation

            pose = pose.inverse()
            pose = torch.mm(v2cam, pose)

            save_pose_path = os.path.join(save_pose_folder, img_path.split("/")[-1][:-4]+'.npy')
            np.save(save_pose_path, pose.cpu().numpy())

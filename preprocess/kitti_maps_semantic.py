import argparse
import os
import sys

sys.path.append("..")
sys.path.append(".")

import h5py
import numpy as np
import pykitti
import torch
import open3d as o3
from tqdm.rich import tqdm

color_map = torch.zeros((260, 3), dtype=torch.int)
color_map[0] = torch.tensor([0, 0, 0])
color_map[1] = torch.tensor([0, 0, 255])
color_map[10] = torch.tensor([245, 150, 100])
color_map[11] = torch.tensor([245, 230, 100])
color_map[13] = torch.tensor([250, 80, 100])
color_map[15] = torch.tensor([150, 60, 30])
color_map[16] = torch.tensor([255, 0, 0])
color_map[18] = torch.tensor([180, 30, 80])
color_map[20] = torch.tensor([255, 0, 0])
color_map[30] = torch.tensor([30, 30, 255])
color_map[31] = torch.tensor([200, 40, 255])
color_map[32] = torch.tensor([90, 30, 150])
color_map[40] = torch.tensor([255, 0, 255])
color_map[44] = torch.tensor([255, 150, 255])
color_map[48] = torch.tensor([75, 0, 75])
color_map[49] = torch.tensor([75, 0, 175])
color_map[50] = torch.tensor([0, 200, 255])
color_map[51] = torch.tensor([50, 120, 255])
color_map[52] = torch.tensor([0, 150, 255])
color_map[60] = torch.tensor([170, 255, 150])
color_map[70] = torch.tensor([0, 175, 0])
color_map[71] = torch.tensor([0, 60, 135])
color_map[72] = torch.tensor([80, 240, 150])
color_map[80] = torch.tensor([150, 240, 255])
color_map[81] = torch.tensor([0, 0, 255])
color_map[99] = torch.tensor([255, 255, 50])
color_map[252] = torch.tensor([245, 150, 100])
color_map[256] = torch.tensor([255, 0, 0])
color_map[253] = torch.tensor([200, 40, 255])
color_map[254] = torch.tensor([30, 30, 255])
color_map[255] = torch.tensor([90, 30, 150])
color_map[257] = torch.tensor([250, 80, 100])
color_map[258] = torch.tensor([180, 30, 80])
color_map[259] = torch.tensor([255, 0, 0])

label_remap = torch.zeros(260, dtype=torch.int)
label_remap[0] = 0  # "unlabeled"
label_remap[1] = 0  # "outlier" mapped to "unlabeled" --------------------------mapped
label_remap[10] = 1  # "car"
label_remap[11] = 2  # "bicycle"
label_remap[13] = 5  # "bus" mapped to "other-vehicle" --------------------------mapped
label_remap[15] = 3  # "motorcycle"
label_remap[16] = 5  # "on-rails" mapped to "other-vehicle" ---------------------mapped
label_remap[18] = 4  # "truck"
label_remap[20] = 5  # "other-vehicle"
label_remap[30] = 6  # "person"
label_remap[31] = 7  # "bicyclist"
label_remap[32] = 8  # "motorcyclist"
label_remap[40] = 9  # "road"
label_remap[44] = 10  # "parking"
label_remap[48] = 11  # "sidewalk"
label_remap[49] = 12  # "other-ground"
label_remap[50] = 13  # "building"
label_remap[51] = 14  # "fence"
label_remap[52] = 0  # "other-structure" mapped to "unlabeled" ------------------mapped
label_remap[60] = 9  # "lane-marking" to "road" ---------------------------------mapped
label_remap[70] = 15  # "vegetation"
label_remap[71] = 16  # "trunk"
label_remap[72] = 17  # "terrain"
label_remap[80] = 18  # "pole"
label_remap[81] = 19  # "traffic-sign"
label_remap[99] = 0  # "other-object" to "unlabeled" ----------------------------mapped
label_remap[252] = 1  # "moving-car" to "car" ------------------------------------mapped
label_remap[253] = 7  # "moving-bicyclist" to "bicyclist" ------------------------mapped
label_remap[254] = 6  # "moving-person" to "person" ------------------------------mapped
label_remap[255] = 8  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
label_remap[256] = 5  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
label_remap[257] = 5  # "moving-bus" mapped to "other-vehicle" -------------------mapped
label_remap[258] = 4  # "moving-truck" to "truck" --------------------------------mapped
label_remap[259] = 5  # "moving-other"-vehicle to "other-vehicle" ----------------mapped

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', default='00',
                    help='sequence')
parser.add_argument('--device', default='cuda',
                    help='device')
parser.add_argument('--voxel_size', default=0.1, help='Voxel Size')
parser.add_argument('--start', default=0, help='Starting Frame')
parser.add_argument('--end', default=100000, help='End Frame')
parser.add_argument('--map', default=None, help='Use map file')
parser.add_argument('--base_folder', default='/data/KITTI/', help='Base odometry folder')

args = parser.parse_args()
sequence = args.sequence
print("Sequnce: ", sequence)
velodyne_folder = os.path.join(args.base_folder, 'sequences', sequence, 'velodyne')
pose_file = os.path.join(args.base_folder, 'sequences', sequence, 'poses.txt')
calib_file = os.path.join(args.base_folder, 'sequences', sequence, 'calib.txt')
kitti = pykitti.odometry(args.base_folder, sequence)

cam0_to_velo = torch.from_numpy(kitti.calib.T_cam0_velo).double()
poses = []
with open(pose_file, 'r') as f:
    for x in f:
        x = x.strip().split()
        x = [float(v) for v in x]
        pose = torch.zeros((4, 4), dtype=torch.float64)
        pose[0, 0:4] = torch.tensor(x[0:4])
        pose[1, 0:4] = torch.tensor(x[4:8])
        pose[2, 0:4] = torch.tensor(x[8:12])
        pose[3, 3] = 1.0
        pose = cam0_to_velo.inverse() @ (pose @ cam0_to_velo)
        poses.append(pose.float())

map_file = args.map
first_frame = int(args.start)
last_frame = min(len(poses), int(args.end))

if map_file is None:

    pc_map = []
    pcl = o3.geometry.PointCloud()
    for i in tqdm(range(first_frame, last_frame)):
        pc = kitti.get_velo(i)
        label_filename = os.path.join(args.base_folder, 'sequences', sequence, "labels", f'{i:06d}.label')
        labels = np.fromfile(label_filename, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF
        labels = torch.tensor(labels.astype(int))
        labels = label_remap[labels]

        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        labels = labels[valid_indices]

        valid_labels_indices = labels > 8
        pc = pc[valid_labels_indices].copy()
        labels = labels[valid_labels_indices]

        pc_color = color_map[labels.long()]

        intensity = pc[:, 3].copy()
        pc[:, 3] = 1.
        RT = poses[i].numpy()
        pc_rot = np.matmul(RT, pc.T)
        pc_rot = pc_rot.astype(float).T.copy()

        pcl_local = o3.geometry.PointCloud()
        pcl_local.points = o3.utility.Vector3dVector(pc_rot[:, :3])
        pcl_local.colors = o3.utility.Vector3dVector(np.vstack((intensity, intensity, intensity)).T)

        downpcd = pcl_local.voxel_down_sample(voxel_size=args.voxel_size)

        pcl.points.extend(downpcd.points)
        pcl.colors.extend(downpcd.colors)

    downpcd_full = pcl.voxel_down_sample(voxel_size=args.voxel_size)
    downpcd, ind = downpcd_full.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)
    o3.io.write_point_cloud(os.path.join(args.base_folder, 'sequences', sequence,
                                         f'semantickitti-{sequence}_{first_frame}-{last_frame}.pcd'), downpcd)
else:
    downpcd = o3.io.read_point_cloud(os.path.join(args.base_folder, 'sequences', sequence, map_file))

voxelized = torch.tensor(downpcd.points, dtype=torch.float)
voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
voxelized = voxelized.t()
voxelized = voxelized.to(args.device)
vox_intensity = torch.tensor(downpcd.colors, dtype=torch.float)[:, 0:1].t().to(args.device)

cam2_to_velo = torch.from_numpy(kitti.calib.T_cam2_velo).float().to(args.device)

# SAVE SINGLE PCs
if not os.path.exists(os.path.join(args.base_folder, 'sequences', sequence,
                                   f'local_maps_semantickitti')):
    os.mkdir(os.path.join(args.base_folder, 'sequences', sequence, f'local_maps_semantickitti'))
for i in tqdm(range(first_frame, last_frame)):
    pose = poses[i]
    pose = pose.to(args.device)
    pose = pose.inverse()

    local_map = voxelized.clone()
    local_intensity = vox_intensity.clone()
    local_map = torch.mm(pose, local_map).t()
    indexes = local_map[:, 1] > -25.
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)
    local_map = local_map[indexes]
    local_intensity = local_intensity[:, indexes]

    local_map = torch.mm(cam2_to_velo, local_map.t())

    file = os.path.join(args.base_folder, 'sequences', sequence,
                        f'local_maps_semantickitti', f'{i:06d}.h5')
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('PC', data=local_map.cpu().half(), compression='lzf', shuffle=True)
        hf.create_dataset('intensity', data=local_intensity.cpu().half(), compression='lzf', shuffle=True)

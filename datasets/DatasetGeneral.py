import csv
import os
from math import radians

import cv2
import h5py
import kornia
import numpy as np
import pykitti
from PIL import Image

import mathutils
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
import visibility
from camera_model import CameraModel
from skimage import io
# from scipy import stats
from torch.utils.data import Dataset
from torchvision import transforms
from utils import invert_pose, rotate_forward, overlay_imgs, merge_inputs


def is_image(img):
    extensions = ['.jpg', '.png', '.tiff', '.jpeg', '.bmp']
    return os.path.splitext(img)[1] in extensions


def get_scan_kitti(pc_path, use_reflectance):
    split_path = pc_path.split('/')
    base_folder = os.path.join('/', *split_path[:-4])
    kitti = pykitti.odometry(base_folder, split_path[-3])
    calib = kitti.calib.K_cam2
    calib = torch.tensor([calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]]).float()

    reflectance = None
    try:
        with h5py.File(pc_path, 'r') as hf:
            pc = hf['PC'][:]
            if use_reflectance:
                reflectance = hf['intensity'][:]
                reflectance = torch.from_numpy(reflectance).float()
    except Exception as e:
        print(f'File Broken: {pc_path}')
        raise e

    return pc, reflectance, calib


class DatasetGeneral(Dataset):

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, maps_folder='point_cloud', change_frame=False,
                 prob_no_aug=0.0, image_folder='image'):
        super(DatasetGeneral, self).__init__()
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.maps_folder = maps_folder
        self.prob_no_aug = prob_no_aug
        self.image_folder = image_folder
        self.change_frame = change_frame

        self.all_files = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:

            img_folder = os.path.join(directory, image_folder)
            point_cloud_folder = os.path.join(directory, self.maps_folder)

            sorted_filenames = sorted(os.listdir(img_folder))
            for filename in sorted_filenames:
                filename_no_extension = os.path.splitext(filename)[0]
                point_cloud_path = os.path.join(point_cloud_folder, filename_no_extension + '.h5')
                if not os.path.exists(point_cloud_path):
                    continue
                self.all_files.append(os.path.join(img_folder, filename))

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        extension = os.path.basename(img_path)
        extension = os.path.splitext(extension)[1]
        pc_path = img_path.replace(f'/{self.image_folder}/', f'/{self.maps_folder}/').replace(extension, '.h5')

        pc, reflectance, calib = get_scan_kitti(pc_path, self.use_reflectance)

        pc_in = torch.from_numpy(pc.astype(np.float32))
        if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
            pc_in = pc_in.t()
        if pc_in.shape[0] == 3:
            homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
            pc_in = torch.cat((pc_in, homogeneous), 0)
        elif pc_in.shape[0] == 4:
            if not torch.all(pc_in[3,:] == 1.):
                pc_in[3,:] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        img = Image.open(img_path)
        do_aug = np.random.rand() > self.prob_no_aug
        h_mirror = False
        if np.random.rand() > 0.5 and self.train and do_aug:
            h_mirror = True
            pc_in[1, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train and do_aug:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.train and do_aug:
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        max_angle = self.max_r
        max_transl = self.max_t

        if do_aug or (not self.train):
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-max_transl, max_transl)
            transl_y = np.random.uniform(-max_transl, max_transl)
            transl_z = np.random.uniform(-max_transl, min(max_transl, 1.))
        else:
            rotz = 0.
            roty = 0.
            rotx = 0.
            transl_x = 0.
            transl_y = 0.
            transl_z = 0.

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'rgb_name': img_path, 'idx': idx}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample

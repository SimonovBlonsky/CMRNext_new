import argparse
import math
import os
import random
import time

import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
import visibility
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich import box
from torch import nn
from tqdm import tqdm

from datasets.DatasetGeneral import DatasetGeneral, DatasetGeneralSingleMapPerSequence
from camera_model import CameraModel
from evaluate_flow_calibration import prepare_input

from models.get_model import get_model
from quaternion_distances import quaternion_loss

from utils import downsample_depth, merge_inputs, get_flow_zforward, quat2mat, tvector2mat, quaternion_from_matrix, \
    EndPointError, rotate_forward, voxelize_gpu, str2bool

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, start_seed):
    seed = start_seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def quaternion_distance(q, r):
    # rinv = r.clone()
    # rinv[1:] *= -1
    # t = rinv[0] * q[0] - rinv[1] * q[1] - rinv[2] * q[2] - rinv[3] * q[3]
    # dist = 2 * math.acos(np.clip(math.fabs(t.item()), 0., 1.))

    dist = quaternion_loss(q.unsqueeze(0), r.unsqueeze(0), q.device)

    dist = 180. * dist.item() / math.pi
    return dist


# noinspection PyUnreachableCode
# @ex.automain
def evaluate_localization(_config, seed):
    global EPOCH

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    cv2.setRNGSeed(seed)
    if _config['deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(os.path.join(_config['weights'][0]), map_location='cpu')

    _config['network'] = checkpoint['config']['network']
    _config['use_reflectance'] = checkpoint['config']['use_reflectance']
    _config['initial_pool'] = checkpoint['config']['initial_pool']
    _config['upsample_method'] = checkpoint['config']['upsample_method']
    _config['occlusion_kernel'] = checkpoint['config']['occlusion_kernel']
    _config['occlusion_threshold'] = checkpoint['config']['occlusion_threshold']
    _config['amp'] = checkpoint['config']['amp']
    _config['scaled_gt'] = checkpoint['config']['scaled_gt']
    _config['uncertainty'] = False
    _config['max_depth'] = checkpoint['config']['max_depth']
    _config['normalize_images'] = checkpoint['config']['normalize_images']
    _config['car_mask'] = False  # Remove this parameter
    _config['labels'] = False  # Remove this parameter
    if 'uncertainty' in checkpoint['config']:
        _config['uncertainty'] = checkpoint['config']['uncertainty']
    _config['fourier_levels'] = -1
    if 'fourier_levels' in checkpoint['config']:
        _config['fourier_levels'] = checkpoint['config']['fourier_levels']
    # _config['flow_direction'] = checkpoint['config']['flow_direction']
    _config['num_scales'] = 1
    if 'num_scales' in checkpoint['config']:
        _config['num_scales'] = checkpoint['config']['num_scales']
    _config['der_type'] = 'NLL'
    if 'der_type' in checkpoint['config']:
        _config['der_type'] = checkpoint['config']['der_type']
    _config['unc_freeze'] = False
    if 'unc_freeze' in checkpoint['config']:
        _config['unc_freeze'] = checkpoint['config']['unc_freeze']
    _config['voxelize'] = 0.1
    if 'voxelize' in checkpoint['config']:
        _config['voxelize'] = checkpoint['config']['voxelize']
    _config['context_encoder'] = 'rgb'
    if 'context_encoder' in checkpoint['config']:
        _config['context_encoder'] = checkpoint['config']['context_encoder']

    mean_torch = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std_torch = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Setup Dataset
    test_directories = []
    base_dir = _config['data_folder']
    if _config['dataset'] == 'argoverse':
        img_shape = (640, 1920//2)  # Multiple of 64, inference at scale 0.5
        for subdir in ['train4']:
            # for subsubdir in os.listdir(os.path.join(base_dir, subdir)):
            for subsubdir in ['15c802a9-0f0e-3c87-b516-a3fa02f1ecb0', '2c07fcda-6671-3ac0-ac23-4a232e0e031e',
                              '91326240-9132-9132-9132-591327440896']:
                test_directories.append(os.path.join(base_dir, subdir, subsubdir))
    elif _config['dataset'] == 'kitti':
        img_shape = (384, 1280)  # Multiple of 64, inference at scale 1
        for subdir in ['00']:
            test_directories.append(os.path.join(base_dir, subdir))
    elif _config['dataset'] == 'pandaset':
        img_shape = (576, 1920//2)  # Multiple of 64 inference at scale 0.5
        for subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:  # Test sequences
            test_directories.append(os.path.join(base_dir, subdir))
    else:
        raise RuntimeError("Dataset unknown")

    if _config['dataset'] == 'pandaset':
        dataset_val = DatasetGeneralSingleMapPerSequence(test_directories, train=False, max_r=_config['max_r'],
                                                         max_t=_config['max_t'],
                                                         use_reflectance=_config['use_reflectance'],
                                                         normalize_images=_config['normalize_images'],
                                                         image_folder='camera/front_camera', dataset='pandaset')
    elif _config['dataset'] == 'argoverse':
        dataset_val = DatasetGeneralSingleMapPerSequence(test_directories, train=False, max_r=_config['max_r'],
                                                         max_t=_config['max_t'],
                                                         use_reflectance=_config['use_reflectance'],
                                                         normalize_images=_config['normalize_images'],
                                                         image_folder='ring_front_center', dataset='argoverse')
    else:
        dataset_val = DatasetGeneral(test_directories, train=False, max_r=_config['max_r'], max_t=_config['max_t'],
                                     use_reflectance=_config['use_reflectance'], normalize_images=_config['normalize_images'],
                                     maps_folder=_config['maps_folder'], image_folder='image_2')


    def init_fn(x):
        return _init_fn(x, seed)

    # Training and test set creation
    num_worker = _config['num_worker']
    batch_size = 1  # This code is designed for a batch size of 1, don't change this!

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = []
    print(f"Loading weights from {_config['weights']}")
    for i in range(len(_config['weights'])):
        checkpoint = torch.load(os.path.join(_config['weights'][i]), map_location='cpu')

        model = get_model(_config, img_shape)
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        model.load_state_dict(clean_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    start_full_time = time.time()

    errors_r = []
    errors_t = []
    list_quats = []
    list_transl = []
    epe = []
    ransac_time = []
    inference_time = []
    for i in range(len(_config['weights'])+1):
        errors_r.append([])
        errors_t.append([])
        list_quats.append([])
        list_transl.append([])
        epe.append([])
    local_loss = 0.0
    tbar = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(tbar):
        # if batch_idx == 100:
        #     break
        start_time = time.time()
        lidar_input = []
        rgb_input = []

        target_flow2 = []
        target_mask2 = []

        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            errors_r[0].append(quaternion_distance(sample['rot_error'][idx],
                                                   torch.tensor([1., 0., 0., 0.], device=sample['rot_error'].device)))
            errors_t[0].append(sample['tr_error'][idx].norm().item())
            list_quats[0].append(sample['rot_error'][idx].cpu().numpy())
            list_transl[0].append(sample['tr_error'][idx].cpu().numpy())

            # ProjectPointCloud in RT-pose

            real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            if _config['max_depth'] < 100.:
                sample['point_cloud'][idx] = sample['point_cloud'][idx][:, sample['point_cloud'][idx][0, :] < _config['max_depth']]

            if _config['voxelize'] != 0.1:
                    sample['point_cloud'][idx] = voxelize_gpu(sample['point_cloud'][idx].T, _config['voxelize']).T.detach()

            pc_rotated = sample['point_cloud'][idx].clone()
            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT1_inv = torch.mm(T, R)
            RTs = [RT1_inv.clone().inverse()]

            pc_rotated = rotate_forward(pc_rotated, RTs[0])

            # Project point cloud into virtual image plane placed at random 'H_init'
            cam_params = sample['calib'][idx].cuda()
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]

            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, pc_rotated, real_shape, reflectance,
                                                                       _config, change_frame=False)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], RT1_inv, cam_model,
                                            [real_shape[0] // 64 * 64, real_shape[1], 3],
                                            scale_flow=False, reverse=False,
                                            get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()

            # Normalize image
            rgb = rgb / 255.
            if _config['normalize_images']:
                rgb = (rgb-mean_torch)/std_torch
            rgb = rgb.permute(2, 0, 1)
            sample['rgb'][idx] = rgb

            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            # flow_mask containts 1 in pixels that have a point projected
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = torch.ones(uv.shape[0], device='cuda', dtype=torch.int)

            points_3D = points_3D[new_indexes].clone()
            shape_pad = [0, 0, 0, 0]

            if _config['dataset'] in ['argoverse', 'pandaset']:
                rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
                depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1,2,0).contiguous(), 2)
                depth_img_no_occlusion = depth_img_no_occlusion.permute(2,0,1)

                shape_pad[3] = (img_shape[0] - real_shape[0]//2)
                shape_pad[1] = (img_shape[1] - real_shape[1]//2)

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

                shape_pad[3] = (img_shape[0]*2 - real_shape[0])
                shape_pad[1] = (img_shape[1]*2 - real_shape[1])
                flow_img = F.pad(flow_img.permute(2,0,1), shape_pad).permute(1,2,0).contiguous()
                flow_mask = F.pad(flow_mask, shape_pad)

            else:
                shape_pad[3] = (img_shape[0] - real_shape[0])
                shape_pad[1] = (img_shape[1] - real_shape[1])

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
                flow_img = F.pad(flow_img.permute(2,0,1), shape_pad).permute(1,2,0).contiguous()
                flow_mask = F.pad(flow_mask, shape_pad)

            down_flow = torch.zeros((img_shape[0] // 4, img_shape[1] // 4, 2), device='cuda', dtype=torch.float)
            down_flow = visibility.downsample_flow(flow_img.contiguous(), down_flow, img_shape[1] // 4, img_shape[0] // 4, 4)
            down_mask = torch.zeros((img_shape[0] // 4, img_shape[1] // 4), device='cuda', dtype=torch.int)
            down_mask = visibility.downsample_mask(flow_mask.contiguous(), down_mask, img_shape[1] // 4, img_shape[0] // 4, 4)

            if _config['fourier_levels'] >= 0:
                depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
                mask = (depth_img_no_occlusion > 0).clone()
                fourier_feats = []
                for L in range(_config['fourier_levels']):
                    fourier_feat = depth_img_no_occlusion * np.pi * 2**L
                    fourier_feats.append(fourier_feat.sin())
                    fourier_feats.append(fourier_feat.cos())
                depth_img_no_occlusion = torch.stack(fourier_feats+[depth_img_no_occlusion])
                depth_img_no_occlusion = depth_img_no_occlusion*mask.unsqueeze(0)

            rgb_input.append(rgb)
            lidar_input.append(depth_img_no_occlusion)
            target_flow2.append(down_flow.permute(2, 0, 1).clone())
            target_mask2.append(down_mask.repeat(2, 1, 1).float().clone())

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)

        rotated_point_cloud = pc_rotated

        for iteration in range(len(_config['weights'])):
            torch.cuda.synchronize()
            time1 = time.time()
            # Predict 'flow': dense lidar depth map to rgb pixel displacements
            with torch.no_grad():
                predicted_flow = models[iteration](rgb_input, lidar_input, iters=24)
                torch.cuda.synchronize()
                time2 = time.time()
                predicted_flow, predicted_uncertainty = predicted_flow

                # print("Network Inference: ", time2 - time1)
                inference_time.append(time2 - time1)
                if _config['dataset'] in ['argoverse', 'pandaset']:
                    predicted_flow = list(predicted_flow)
                    for scale in range(len(predicted_flow)):
                        predicted_flow[scale] *= 2
                        predicted_flow[scale] = F.interpolate(predicted_flow[scale], scale_factor=2, mode='bilinear')
                        if _config['uncertainty']:
                            predicted_uncertainty[scale] = F.interpolate(predicted_uncertainty[scale], scale_factor=2,
                                                                         mode='bilinear')

            up_flow = predicted_flow[-1]

            # EPE
            gt = flow_img.clone().permute(2,0,1)
            gt = torch.cat((gt, flow_mask.unsqueeze(0).float()))
            gt = gt.unsqueeze(0)
            epe[iteration].append(EndPointError(up_flow, gt).item())

            up_flow = up_flow[0].permute(1, 2, 0)

            new_uv = uv.float() + up_flow[uv[:, 1], uv[:, 0]]

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            # Check only pixels that are within the image border
            valid_indexes = valid_indexes & (new_uv[:,0] < flow_mask.shape[1])
            valid_indexes = valid_indexes & (new_uv[:,1] < flow_mask.shape[0])
            valid_indexes = valid_indexes & (new_uv[:,0] >= 0)
            valid_indexes = valid_indexes & (new_uv[:,1] >= 0)
            new_uv = new_uv[valid_indexes]

            valid_indexes2 = torch.ones(new_uv.shape[0], dtype=torch.bool).cuda()

            new_uv = new_uv[valid_indexes2]

            points_2d = new_uv.cpu().numpy()
            obj_coord = points_3D[valid_indexes][valid_indexes2][:, :3].cpu().numpy()
            obj_coord_zforward = np.zeros(obj_coord.shape)
            obj_coord_zforward[:, 0] = obj_coord[:, 0]
            obj_coord_zforward[:, 1] = obj_coord[:, 1]
            obj_coord_zforward[:, 2] = obj_coord[:, 2]
            cam_mat = cam_model.get_matrix()

            torch.cuda.synchronize()

            time1 = time.time()
            if obj_coord_zforward.shape[0] < 10:
                for asd in range(iteration+1):
                    errors_t[asd].pop(-1)
                    errors_r[asd].pop(-1)
                    list_transl[asd].pop(-1)
                    list_quats[asd].pop(-1)
                break

            cuda_pnp = cv2.pythoncuda.cudaPnP(obj_coord_zforward.astype(np.float32).copy(),
                                              points_2d.astype(np.float32).copy(), obj_coord_zforward.shape[0],
                                              200, 2.,cam_mat.astype(np.float32)
                                              )

            transl = cuda_pnp[0, [0, 1, 2]]
            rot_mat = cuda_pnp[:, 3:6].T
            rot_mat, _ = cv2.Rodrigues(rot_mat)

            torch.cuda.synchronize()
            time2 = time.time()
            # print("Ransac: ", time2-time1)
            ransac_time.append(time2 - time1)

            transl = torch.tensor(transl).float().squeeze().cuda()

            rot_mat = torch.tensor(rot_mat)

            pred_quaternion = quaternion_from_matrix(rot_mat)
            # pred_quaternion = pred_quaternion[[0, 3, 1, 2]]  # Z-forward to X-forward

            R_predicted = quat2mat(pred_quaternion).cuda()
            T_predicted = tvector2mat(transl)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            composed = torch.mm(RTs[iteration], RT_predicted.inverse())
            RTs.append(composed)

            T_composed = composed[:3, 3]
            R_composed = quaternion_from_matrix(composed)
            errors_r[iteration+1].append(quaternion_distance(R_composed,
                                                           torch.tensor([1., 0., 0., 0.], device=R_composed.device)))
            errors_t[iteration+1].append(T_composed.norm().item())

            list_transl[iteration+1].append(T_composed.cpu().numpy())
            list_quats[iteration+1].append(R_composed.cpu().numpy())

            if T_composed.norm().item() > 4.:
                for left_iteration in range(iteration+2, len(_config['weights'])+1):
                    errors_t[left_iteration].append(T_composed.norm().item())
                    errors_r[left_iteration].append(errors_r[iteration+1][-1])
                break

            # Rotate point cloud based on predicted pose, and generate new
            # inputs for the next iteration
            rotated_point_cloud = rotate_forward(sample['point_cloud'][idx], RTs[-1])

            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, rotated_point_cloud, real_shape,
                                                                       reflectance, _config, change_frame=False)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth[indexes], RTs[-1].inverse(), cam_model,
                                                    [real_shape[0], real_shape[1], 3],
                                                    scale_flow=False, reverse=False,
                                                    get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()
            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            points_3D = points_3D[new_indexes].clone()
            shape_pad = [0, 0, 0, 0]

            if _config['dataset'] in ['argoverse', 'pandaset']:
                original_img = rgb.clone()
                rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
                depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1,2,0).contiguous(), 2)
                depth_img_no_occlusion = depth_img_no_occlusion.permute(2,0,1)

                shape_pad[3] = (img_shape[0] - real_shape[0]//2)
                shape_pad[1] = (img_shape[1] - real_shape[1]//2)

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

                shape_pad[3] = (img_shape[0]*2 - real_shape[0])
                shape_pad[1] = (img_shape[1]*2 - real_shape[1])
                flow_img = F.pad(flow_img.permute(2,0,1), shape_pad).permute(1,2,0).contiguous()
                flow_mask = F.pad(flow_mask, shape_pad)

            else:
                shape_pad[3] = (img_shape[0] - real_shape[0])
                shape_pad[1] = (img_shape[1] - real_shape[1])

                rgb = F.pad(rgb, shape_pad)
                depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
                flow_img = F.pad(flow_img.permute(2,0,1), shape_pad).permute(1,2,0).contiguous()
                flow_mask = F.pad(flow_mask, shape_pad)

            if _config['fourier_levels'] >= 0:
                depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
                mask = (depth_img_no_occlusion > 0).clone()
                fourier_feats = []
                for L in range(_config['fourier_levels']):
                    fourier_feat = depth_img_no_occlusion * np.pi * 2**L
                    fourier_feats.append(fourier_feat.sin())
                    fourier_feats.append(fourier_feat.cos())
                depth_img_no_occlusion = torch.stack(fourier_feats+[depth_img_no_occlusion])
                depth_img_no_occlusion = depth_img_no_occlusion*mask.unsqueeze(0)

            rgb_input = rgb.unsqueeze(0)
            lidar_input = depth_img_no_occlusion.unsqueeze(0)

            down_flow = torch.zeros((img_shape[0] // 4, img_shape[1] // 4, 2), device='cuda', dtype=torch.float)
            down_flow = visibility.downsample_flow(flow_img, down_flow, img_shape[1] // 4, img_shape[0] // 4, 4)
            down_mask = torch.zeros((img_shape[0] // 4, img_shape[1] // 4), device='cuda', dtype=torch.int)
            down_mask = visibility.downsample_mask(flow_mask, down_mask, img_shape[1] // 4, img_shape[0] // 4, 4)

            try:
                tbar.set_postfix(t_mean=torch.tensor(errors_t[-1]).mean().item(),
                                 t_median=torch.tensor(errors_t[-1]).median().item(),
                                 r_mean=torch.tensor(errors_r[-1]).mean().item(),
                                 r_median=torch.tensor(errors_r[-1]).median().item(),
                                 epe_mean=torch.tensor(epe[0]).mean().item())
            except:
                pass

    errors_t = torch.tensor(errors_t)
    errors_r = torch.tensor(errors_r)
    print(f'Network Time: ', torch.tensor(inference_time).mean())
    print(f'PnP+RANSAC Time: ', torch.tensor(ransac_time).mean())
    console = Console()
    table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
    table.title = f"CMRNext Results on {_config['dataset']}"
    table.add_column("Iteration")
    table.add_column("Median Translation error (cm)", justify="center", max_width=20)
    table.add_column("Median Rotation error (˚)", justify="center", max_width=20)
    table.add_row(
        f"Initial Pose",
        f"{errors_t[0].median().item() * 100:.2f}",
        f"{errors_r[0].median().item():.2f}"
    )
    for iteration in range(1, len(_config['weights']) + 1):
        table.add_row(
            f"Iteration {iteration}",
            f"{errors_t[iteration].median().item() * 100:.2f}",
            f"{errors_r[iteration].median().item():.2f}"
        )
    print("")
    print("")
    console.print(table)
    if _config['save_file'] is not None:
        torch.save(errors_t, f'./{_config["save_file"]}_errors_t.torch')
        torch.save(errors_r, f'./{_config["save_file"]}_errors_r.torch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--data_folder', type=str, default='/data/KITTI/sequences/')
    parser.add_argument('--max_t', type=float, default=2)
    parser.add_argument('--max_r', type=float, default=10.)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--weights', type=str, nargs='+', default=None)
    parser.add_argument('--img_shape', type=int, nargs=1, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_file', type=str, nargs='?', default=None)
    parser.add_argument('--quantile', type=float, default=1.0)
    parser.add_argument('--maps_folder', type=str, default="local_maps_semantickitti")

    args = parser.parse_args()
    _config = vars(args)
    evaluate_localization(_config, _config['seed'])


if __name__ == '__main__':
    main()

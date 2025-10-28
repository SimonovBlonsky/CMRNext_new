python3 train_calibration.py  --savemodel checkpoints/iter1/ \
    --data_folder_argo /mnt/nas_9/datasets/Argoverse1/argoverse-tracking/ \
    --data_folder_kitti /mnt/nas_9/datasets/kitti_odometry/dataset/sequences/ \
    --data_folder_panda /mnt/nas_9/datasets/pandaset/PandaSet/ \
    --max_r 20 --max_t 1.5

python3 train_calibration.py  --savemodel checkpoints/iter2/ \
    --data_folder_argo /mnt/nas_9/datasets/Argoverse1/argoverse-tracking/ \
    --data_folder_kitti /mnt/nas_9/datasets/kitti_odometry/dataset/sequences/ \
    --data_folder_panda /mnt/nas_9/datasets/pandaset/PandaSet/ \
    --max_r 1 --max_t 0.1

python3 train_calibration.py  --savemodel checkpoints/iter3/ \
    --data_folder_argo /mnt/nas_9/datasets/Argoverse1/argoverse-tracking/ \
    --data_folder_kitti /mnt/nas_9/datasets/kitti_odometry/dataset/sequences/ \
    --data_folder_panda /mnt/nas_9/datasets/pandaset/PandaSet/ \
    --max_r 0.2 --max_t 0.05
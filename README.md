<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <h1>CMRNext: Camera to LiDAR Matching in the Wild for<br/>Localization and Extrinsic Calibration</h1>
    <a href="https://cmrnext.cs.uni-freiburg.de/" target="_blank" style="margin: 2px;"><img alt="Project Website"
    src="https://img.shields.io/badge/🌐%20Project Website-CMRNext-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/robot-learning-freiburg/CMRNext/" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-CMRNext-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://arxiv.org/abs/2402.00129" target="_blank" style="margin: 2px;"><img src="https://img.shields.io/badge/arXiv-2402.00129-b31b1b" alt="arXiv"  style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://ieeexplore.ieee.org/document/10908048" target="_blank" style="margin: 2px;"><img src="https://img.shields.io/badge/IEEE-Xplore-blue" alt="arXiv" style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://github.com/robot-learning-freiburg/CMRNext/blob/main/LICENSE" target="_blank" style="margin: 2px;"><img src="https://img.shields.io/github/license/robot-learning-freiburg/CMRNext" style="display: inline-block; vertical-align: middle;" /></a>
</div>


Repository providing the source code for the paper
>CMRNext: Camera to LiDAR Matching in the Wild for Localization and Extrinsic Calibration
>
>[Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo) and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)  
>IEEE Transactions on Robotics, 2025.  

<p align="center">
  <img src="assets/overview.png" alt="Overview of Teleop" width="1200" />
</p>

If you use CMRNext, please cite:
```
@article{cattaneo2025cmrnext,
    title={CMRNext: Camera to LiDAR Matching in the Wild for Localization and Extrinsic Calibration}, 
    author={Daniele Cattaneo and Abhinav Valada},
    journal={IEEE Transactions on Robotics}, 
    year={2025},
    volume={41},
    pages={1995-2013},
    doi={10.1109/TRO.2025.3546784}
}
```

## Release Status
- [x] 2025/03/25 We released the ROS code for camera-LiDAR extrinsic calibration on KITTI using [MDPCalib](https://github.com/robot-learning-freiburg/MDPCalib/), which combines CMRNext with graph optimization.
- [x] 2025/03/26 - We released the [inference code for extrinsic calibration](#Camera-LiDAR-Extrinsic-Calibration)
- [ ] Localization inference code
- [ ] Training code


## Abstract
LiDARs are widely used for mapping and localization in dynamic environments. However, their high cost limits their widespread adoption. On the other hand, monocular localization in LiDAR maps using inexpensive cameras is a cost-effective alternative for large-scale deployment. Nevertheless, most existing approaches struggle to generalize to new sensor setups and environments, requiring retraining or fine-tuning. In this paper, we present CMRNext, a novel approach for camera-LIDAR matching that is independent of sensor-specific parameters, generalizable, and can be used in the wild for monocular localization in LiDAR maps and camera-LiDAR extrinsic calibration. CMRNext exploits recent advances in deep neural networks for matching cross-modal data and standard geometric techniques for robust pose estimation. We reformulate the point-pixel matching problem as an optical flow estimation problem and solve the Perspective-n-Point problem based on the resulting correspondences to find the relative pose between the camera and the LiDAR point cloud. We extensively evaluate CMRNext on six different robotic platforms, including three publicly available datasets and three in-house robots. Our experimental evaluations demonstrate that CMRNext outperforms existing approaches on both tasks and effectively generalizes to previously unseen environments and sensor setups in a zero-shot manner.

## Datasets
The code natively supports the following datasets: [KITTI](#kitti), [Argoverse 1](#argoverse), and [Pandaset](#pandaset).

#### KITTI
Download the `KITTI Odometry` dataset available [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
All parts except for the grayscale images are required. Additionally, download the ground truth poses from [SemanticKITTI](https://semantic-kitti.org/dataset.html#download), specifically the `SemanticKITTI label data`.

After extracting all files (e.g., in the folder `\data\KITTI`), the folder structure should look like
```bash
\data\KITTI\sequences
├── 00
│   ├── image_2
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── ...
│   │   └── 004540.png
│   ├── image_3
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   ├── ...
│   │   └── 004540.png
│   ├── velodyne
│   │   ├── 000000.bin
│   │   ├── 000001.bin
│   │   ├── ...
│   │   └── 004540.bin
│   └── poses.txt
│   └── calib.txt
└── 01
    ├── ...

```

#### Argoverse
Download the four train sets of `Argoverse 3D Tracking v1.1` from the [official website](https://www.argoverse.org/av1.html#download-link).
Extract all files (e.g., in the folder `\data\argoverse`)

#### Pandaset
Unfortunately, the pandaset dataset is not available for download from their [official website](https://scale.com/resources/download/pandaset) anymore.
An alternative download is available at [Kaggle](https://www.kaggle.com/datasets/pz19930809/pandaset), however I haven't tested it.

## Downloading model weights for extrinsic calibration

Please download the model weights for camera-LiDAR extrinsic calibration from this link and store them under: `/data/`.
- Model weights: https://calibration.cs.uni-freiburg.de/downloads/cmrnext_weights.zip

## Camera-LiDAR Extrinsic Calibration
The best way to run CMRNext is by using Docker. Given the endless variability in system setups, many things can go wrong when configuring the environment manually. If you choose not to use Docker, it may be difficult for me to help troubleshoot any issues you encounter.

### Docker
Install [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Tested with `Docker version 28.0.1` and `NVIDIA Container Toolkit version 1.17.5-1`

> [!CAUTION]
> The provided Docker container does NOT support NVIDIA RTX 50 Series (Blackwell).
> Adding support for it has low priority currently. Feel free to open a pull request.

- To build the image, run `docker build . -t cmrnext` in the root of this repository.
- Prepare using GUIs in the container: `xhost +local:docker`.
- Start container and mount the folder where your dataset are located: 
```bash
docker run --runtime=nvidia -it -v /tmp/.X11-unix:/tmp/.X11-unix -v PATH_TO_DATA:/data -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all cmrnext
```
- Within the container, move to the code folder `cd /root/CMRNext/`

Finally, run the inference, assuming the weights and the datasets are located under `\data\ `, change the paths according to your setup.

KITTI left camera:
```bash
python3 evaluate_flow_calibration.py --weights /data/cmrnext-calib-LEnc-iter1.tar /data/cmrnext-calib-LEnc-iter5.tar /data/cmrnext-calib-LEnc-iter6.tar --data_folder /data/KITTI/sequences/ --dataset kitti
```

KITTI right camera:
```bash
python3 evaluate_flow_calibration.py --weights /data/cmrnext-calib-LEnc-iter1.tar /data/cmrnext-calib-LEnc-iter5.tar /data/cmrnext-calib-LEnc-iter6.tar --data_folder /data/KITTI/sequences/ --dataset kitti --cam 3
```

Argoverse V1:
```bash
python3 evaluate_flow_calibration.py --weights /data/cmrnext-calib-LEnc-iter1.tar /data/cmrnext-calib-LEnc-iter5.tar /data/cmrnext-calib-LEnc-iter6.tar --data_folder /data/argoverse/argoverse-tracking/ --dataset argoverse
```

Pandaset:
```bash
python3 evaluate_flow_calibration.py --weights /data/cmrnext-calib-LEnc-iter1.tar /data/cmrnext-calib-LEnc-iter5.tar /data/cmrnext-calib-LEnc-iter6.tar --data_folder /data/pandaset/ --dataset pandaset
```

## Contacts
* [Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo)
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.

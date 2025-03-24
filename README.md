# CMRNext: Camera to LiDAR Matching in the Wild for Localization and Extrinsic Calibration
[**arXiv**](https://arxiv.org/abs/2402.00129) | [**Project Website**](https://cmrnext.cs.uni-freiburg.de/) | [**IEEEXplore**](https://ieeexplore.ieee.org/document/10908048)

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

## TODO
- [x] We released the ROS code for camera-LiDAR extrinsic calibration on KITTI using [MDPCalib](https://github.com/robot-learning-freiburg/MDPCalib/), which combines CMRNext with graph optimization.
- [ ] Calibration inference code
- [ ] Localization inference code
- [ ] Training code

## Abstract
LiDARs are widely used for mapping and localization in dynamic environments. However, their high cost limits their widespread adoption. On the other hand, monocular localization in LiDAR maps using inexpensive cameras is a cost-effective alternative for large-scale deployment. Nevertheless, most existing approaches struggle to generalize to new sensor setups and environments, requiring retraining or fine-tuning. In this paper, we present CMRNext, a novel approach for camera-LIDAR matching that is independent of sensor-specific parameters, generalizable, and can be used in the wild for monocular localization in LiDAR maps and camera-LiDAR extrinsic calibration. CMRNext exploits recent advances in deep neural networks for matching cross-modal data and standard geometric techniques for robust pose estimation. We reformulate the point-pixel matching problem as an optical flow estimation problem and solve the Perspective-n-Point problem based on the resulting correspondences to find the relative pose between the camera and the LiDAR point cloud. We extensively evaluate CMRNext on six different robotic platforms, including three publicly available datasets and three in-house robots. Our experimental evaluations demonstrate that CMRNext outperforms existing approaches on both tasks and effectively generalizes to previously unseen environments and sensor setups in a zero-shot manner.

## Contacts
* [Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo)
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.

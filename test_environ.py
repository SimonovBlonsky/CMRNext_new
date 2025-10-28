# import argparse
# from functools import reduce

# import cv2
# import mathutils
# import torch
# import torch.nn.functional as F
# import liegroups
# import logging
# import math
# import numpy as np
# import scipy
# import visibility
# from matplotlib import cm
# from torch.utils.data.dataloader import default_collate
from torch_scatter import scatter_mean
# from torch_geometric.nn.pool.consecutive import consecutive_cluster
# from torch_geometric.nn import voxel_grid

# try:
#     import geomstats.geometry.lie_group as lie_group
# except ImportError:
#     import geomstats.lie_group as lie_group
import math

import numpy as np
import torch

def diff1(q, r):
    """
        Distanza che approssima quella esatta usando acos

    """
    dot = np.dot(q, r)
    return 2*np.math.acos(np.math.fabs(dot))


def quatmultiply(q, r, device='cpu'):
    """
    Batch quaternion multiplication
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]
        r (torch.Tensor/np.ndarray): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        t = np.zeros(q.shape[0], 4)
    else:
        raise TypeError("Type not supported")
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t


def quatinv(q):
    """
    Batch quaternion inversion
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]

    Returns:
        torch.Tensor/np.ndarray: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError("Type not supported")
    t *= -1
    t[:, 0] *= -1
    return t


def quaternion_loss(q, r, device):
    """
    Batch quaternion distances, used as loss
    Args:
        q (torch.Tensor): shape=[Nx4]
        r (torch.Tensor): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[N]
    """
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))


def diff3(q, r):
    """
        Distanza Approssimata

    """
    dot = np.dot(q, r)
    return 1. - np.abs(dot)


def rotation_from_quaternion(x):
    """
        Converts a quaternion to a rotation matrix.

    """
    # Documented in <http://en.wikipedia.org/w/index.php?title=
    # Quaternions_and_spatial_rotation&oldid=402924915>
    a, b, c, d = x

    r1 = [a ** 2 + b ** 2 - c ** 2 - d ** 2,
          2 * b * c - 2 * a * d,
          2 * b * d + 2 * a * c]
    r2 = [2 * b * c + 2 * a * d,
          a ** 2 - b ** 2 + c ** 2 - d ** 2,
          2 * c * d - 2 * a * b]
    r3 = [2 * b * d - 2 * a * c,
          2 * c * d + 2 * a * b,
          a ** 2 - b ** 2 - c ** 2 + d ** 2]

    return np.array([r1, r2, r3])


def axis_angle_from_rotation(R):
    '''
        Returns the *(axis,angle)* representation of a given rotation.

        There are a couple of symmetries:

        * By convention, the angle returned is nonnegative.

        * If the angle is 0, any axis will do.
          In that case, :py:func:`default_axis` will be returned.

    '''
    angle = np.arccos(np.clip(((R.trace() - 1) / 2), -1.0, 1.0))

    if angle == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    else:
        v = np.array([R[2, 1] - R[1, 2],
                   R[0, 2] - R[2, 0],
                   R[1, 0] - R[0, 1]])

        computer_with_infinite_precision = False
        if computer_with_infinite_precision:
            axis = (1 / (2 * np.sin(angle))) * v
        else:
            # OK, the formula above gives (theoretically) the correct answer
            # but it is imprecise if angle is small (dividing by a very small
            # quantity). This is way better...
            axis = (v * np.sign(angle)) / np.linalg.norm(v)

        return axis, angle


def geodesic_distance_for_rotations(R1, R2):
    '''
        Returns the geodesic distance between two rotation matrices.

        It is computed as the angle of the rotation :math:`R_1^{*} R_2^{-1}``.

    '''
    R = np.dot(R1, R2.T)
    axis1, angle1 = axis_angle_from_rotation(R)  # @UnusedVariable
    return angle1

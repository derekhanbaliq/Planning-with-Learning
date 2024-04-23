"""
    helper functions for the lidar scan
    Author: Derek Zhou, Biao Wang, Tian Tan
"""
import numpy as np


def downsample_lidar_scan(data, observation_shape):
    # print("observation_shape type: ", type(observation_shape))
    # print("observation_shape: ", observation_shape)
    obs_gap = int(1080 / observation_shape)
    processed_data = data[::obs_gap]

    return processed_data


def get_lidar_data(scans, poses_x, poses_y, poses_theta):
    poses_x = poses_x
    poses_y = poses_y
    poses_theta = poses_theta
    angles = np.linspace(-135, 135, num=scans.shape[0]) * (np.pi / 180)

    # Local coordinates in the LiDAR frame of reference
    local_x = scans * np.cos(angles)
    local_y = scans * np.sin(angles)

    # Rotation and translation to global coordinates
    cos_theta = np.cos(poses_theta)
    sin_theta = np.sin(poses_theta)
    global_x = cos_theta * local_x - sin_theta * local_y + poses_x
    global_y = sin_theta * local_x + cos_theta * local_y + poses_y

    lidar_data = np.vstack((global_x, global_y)).T

    return lidar_data



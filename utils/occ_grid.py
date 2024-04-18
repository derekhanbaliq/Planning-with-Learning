"""
    the occupancy grid & helper functions for the perception module
    Author: Derek Zhou, Biao Wang, Tian Tan
"""

import numpy as np


def downsample_lidar_scan(data, observation_shape, method):
    if method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080 / observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data

    return processed_data


class OccGrid:
    def __init__(self, scans, poses_x, poses_y, poses_theta):
        self.lidar_scan = np.array(scans)  # Lidar scan data
        self.poses_x = poses_x  # x-positions of agents
        self.poses_y = poses_y  # y-positions of agents
        self.poses_theta = poses_theta  # orientations of agents in radians
        self.num_points = self.lidar_scan.shape[1]  # Expected to be 1080
        self.angles = (np.linspace(-135, 135, self.num_points)) * (np.pi / 180)

    def get_OccGrid(self, idx=0):
        # Calculate local coordinates in the LiDAR's frame of reference
        local_x_coords = self.lidar_scan[idx] * np.cos(self.angles)
        local_y_coords = self.lidar_scan[idx] * np.sin(self.angles)

        # Convert local coordinates to global coordinates considering the orientation
        cos_theta = np.cos(self.poses_theta[idx])
        sin_theta = np.sin(self.poses_theta[idx])
        global_x_coords = cos_theta * local_x_coords - sin_theta * local_y_coords + self.poses_x[idx]
        global_y_coords = sin_theta * local_x_coords + cos_theta * local_y_coords + self.poses_y[idx]

        return np.vstack((global_x_coords, global_y_coords)).T
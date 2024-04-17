import numpy as np
import matplotlib.pyplot as plt


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
"""
    helper functions for trajectories
    Author: Derek Zhou, Biao Wang, Tian Tan
"""
import numpy as np
# import matplotlib as plt
class OccGrid:

    def __init__(self,obs):
        self.lidar_scan = obs # 1080*1
        self.num_points = self.lidar_scan.shape[0] # 1080
        self.angles = self.angles = np.linspace(-135, 135, self.num_points) * (np.pi / 180)


    def get_OccGrid(self):
        # calculate x and y coords
        x_coords = self.data * np.cos(self.angles)  # r * cos(θ)
        y_coords = self.data * np.sin(self.angles)  # r * sin(θ)
        return np.vstack((x_coords, y_coords)).T  # 1080 x 2
    # def downsample_lidar_scan(data, observation_shape, method):
    #     if method == "simple":
    #         # print("observation_shape type: ", type(observation_shape))
    #         # print("observation_shape: ", observation_shape)
    #         obs_gap = int(1080 / observation_shape)
    #         processed_data = data[::obs_gap]
    #     else:
    #         processed_data = data
    #
    #     return processed_data



# obs =
# # data = np.random.rand(1080, 1) * 100
# grid = OccGrid(data)
# xy_coords = grid.get_OccGrid()
# # print(xy_coords)
# print(data.shape[0])


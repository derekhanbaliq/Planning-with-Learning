"""
    helper functions for trajectories
    Author: Derek Zhou, Biao Wang, Tian Tan
"""


def downsample_lidar_scan(data, observation_shape, method):
    if method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080 / observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data

    return processed_data






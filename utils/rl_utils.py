import numpy as np
from scipy.spatial import distance


def downsample(data, observation_shape, downsampling_method):
    if downsampling_method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080 / observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data

    return processed_data


def get_ref_traj_in_horizon(obs, waypoints, predict_time=2):
    waypoints = np.array([waypoints.x, waypoints.y]).T
    num_waypoints = waypoints.shape[0]

    cur_x = obs['poses_x'][0]  # ego vehicle
    cur_y = obs['poses_y'][0]
    cur_pos = np.array([cur_x, cur_y]).reshape((1, 2))

    distances = distance.cdist(cur_pos, waypoints, 'euclidean').reshape((num_waypoints,))
    closest_index = np.argmin(distances)
    # print(closest_index, waypoints[closest_index])
    predict_dist = obs['linear_vels_x'][0] * predict_time  # use current speed, cuz future movement is subject to change
    # print(predict_dist)

    traj = []
    i = closest_index
    dist = distances[closest_index]  # accumulated distance
    while dist < predict_dist:
        # suppose anti-clockwise
        prev_i = i
        i = 0 if i == num_waypoints - 1 else i + 1
        dist += np.linalg.norm(waypoints[i] - waypoints[prev_i])
        traj.append(np.hstack((i, waypoints[i])))

    traj = np.array(traj)
    print(traj.shape)

    return traj

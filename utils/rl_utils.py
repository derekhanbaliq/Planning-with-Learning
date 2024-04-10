import numpy as np
from scipy.spatial import distance
from scipy.interpolate import interp1d


def downsample(data, observation_shape, downsampling_method):
    if downsampling_method == "simple":
        # print("observation_shape type: ", type(observation_shape))
        # print("observation_shape: ", observation_shape)
        obs_gap = int(1080 / observation_shape)
        processed_data = data[::obs_gap]
    else:
        processed_data = data

    return processed_data


def get_front_traj(obs, profile, predict_time=2):
    waypoints = np.array([profile.x, profile.y]).T
    ref_speed = profile.v
    # ref_curvature = profile.γ
    num_waypoints = waypoints.shape[0]

    cur_x = obs['poses_x'][0]  # ego vehicle
    cur_y = obs['poses_y'][0]
    cur_pos = np.array([cur_x, cur_y]).reshape((1, 2))

    distances = distance.cdist(cur_pos, waypoints, 'euclidean').reshape((num_waypoints,))
    closest_index = np.argmin(distances)
    # print(closest_index, waypoints[closest_index])

    traj = []
    i = closest_index
    t = profile.unit_dist / ref_speed[i]  # accumulated time
    while t < predict_time:
        # suppose anti-clockwise
        i = 0 if i == num_waypoints - 1 else i + 1
        t += profile.unit_dist / ref_speed[i]  # i -> i + 1
        traj.append(np.hstack((i, waypoints[i], ref_speed[i])))

    traj = np.array(traj)

    return traj


def get_interpolated_traj_with_horizon(traj, h):
    len = traj.shape[0]
    steps = np.linspace(start=1, stop=len, num=len, endpoint=False)

    h_traj = []
    for i in range(1, traj.shape[1]):  # x, y, v
        val = traj[:, i]
        interp_func = interp1d(steps, val.flatten(), kind='cubic')
        new_steps = np.linspace(start=1, stop=len, num=h, endpoint=False)
        new_val = interp_func(new_steps)
        h_traj.append(new_val)
    h_traj = np.array(h_traj).T

    return h_traj


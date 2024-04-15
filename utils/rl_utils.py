"""
    main application for RL planner
    Author: Derek Zhou, Biao Wang, Tian Tan
"""

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
    num_points = traj.shape[0]
    # linspace - https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    steps = np.linspace(start=0, stop=num_points, num=num_points, endpoint=True)

    h_traj = []
    for i in range(1, traj.shape[1]):  # x, y, v
        val = traj[:, i]
        interp_func = interp1d(steps, val.flatten(), kind='cubic')
        new_steps = np.linspace(start=0, stop=num_points, num=h, endpoint=True)
        new_val = interp_func(new_steps)
        h_traj.append(new_val)
    h_traj = np.array(h_traj).T

    return h_traj


def densify_offset_traj(offset_traj, intep_num=80):
    num_points = offset_traj.shape[0]  # horizon
    steps = np.linspace(start=0, stop=num_points, num=num_points, endpoint=True)  # index, 0 ~ 10

    profile = np.zeros((intep_num, 3))
    for i in range(3):  # offset_traj.shape[1] = 3, [x, y, v]
        val = offset_traj[:, i]
        interp_func = interp1d(steps, val.flatten(), kind='cubic')
        new_steps = np.linspace(start=0, stop=num_points,
                                num=intep_num, endpoint=False)  # even at 8m/s, the unit step is 0.2m
        profile[:, i] = np.array(interp_func(new_steps))
    # print(profile.shape)

    return profile


import numpy as np


def add_lateral_offset2get_new_traj(traj, offset):
    traj = np.array(traj)
    offset = np.asarray(offset)

    # Extract only the x and y
    xy_traj = traj[:, :2]

    # Initialize tangent vectors in 2D
    tangents = np.zeros_like(xy_traj)

    # Calculate tangents for the trajectory points
    tangents[0] = xy_traj[1] - xy_traj[0]  # tangent at the first point 1
    tangents[-1] = xy_traj[-1] - xy_traj[-2]  # tangent at the last point 10
    tangents[1:-1] = xy_traj[2:] - xy_traj[:-2]  # tangents for the middle points 2-9

    # Normalize tangent vectors to get unit tangent vectors
    norms = np.linalg.norm(tangents, axis=1)
    unit_tangents = tangents / norms[:, np.newaxis]

    # Calculate perpendicular vectors in 2D (rotate unit tangent vectors 90 degrees)
    normals = np.array([-unit_tangents[:, 1], unit_tangents[:, 0]]).T

    # Normalize normals for consistency and safety
    normal_norms = np.linalg.norm(normals, axis=1)
    unit_normals = normals / normal_norms[:, np.newaxis]

    # Calculate new points by adding the scaled normal vectors to original points
    offsets_rescaled = offset[:, np.newaxis] * unit_normals
    new_xy_traj = xy_traj + offsets_rescaled

    # Combine the new x and y coordinates with the original 'v' values
    new_traj = np.hstack([new_xy_traj, traj[:, 2:]])

    return new_traj


"""
    helper functions for trajectories
    Author: Derek Zhou, Biao Wang, Tian Tan
"""

import numpy as np
from scipy.spatial import distance
from scipy.interpolate import interp1d


def get_front_traj(obs, profile, predict_time=2.0):
    waypoints = np.array([profile.x, profile.y]).T
    ref_speed = profile.v
    ref_heading = profile.θ
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
        traj.append(np.hstack((i, waypoints[i], ref_speed[i], ref_heading[i])))

    traj = np.array(traj)

    return traj


def get_interpolated_traj_with_horizon(traj, h):
    num_points = traj.shape[0]
    # linspace - https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    steps = np.linspace(start=0, stop=num_points, num=num_points, endpoint=True)

    h_traj = []
    for i in range(1, traj.shape[1]):  # x, y, v, θ
        val = traj[:, i]
        interp_func = interp1d(steps, val.flatten(), kind='cubic')
        new_steps = np.linspace(start=0, stop=num_points, num=h, endpoint=True)
        new_val = interp_func(new_steps)
        h_traj.append(new_val)
    h_traj = np.array(h_traj).T

    return h_traj


def get_offset_traj(traj, offset):
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


def densify_offset_traj(offset_traj, intep_num=80):
    num_points = offset_traj.shape[0]  # horizon
    steps = np.linspace(start=0, stop=num_points, num=num_points, endpoint=True)  # index, 0 ~ 10

    profile = np.zeros((intep_num, 4))
    for i in range(4):  # offset_traj.shape[1] = 4, [x, y, v, theta]
        val = offset_traj[:, i]
        interp_func = interp1d(steps, val.flatten(), kind='linear')
        new_steps = np.linspace(start=0, stop=num_points,
                                num=intep_num, endpoint=False)  # even at 8m/s, the unit step is 0.2m
        profile[:, i] = np.array(interp_func(new_steps))
    # print(profile.shape)

    return profile


def global_to_local(obs, global_data):
    x, y, theta = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
    H = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                  [np.sin(theta), np.cos(theta), 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # vehicle frame to world frame

    global_coord = np.hstack((global_data[:, :2], np.tile([0, 1], (global_data.shape[0], 1))))
    local_coord = np.transpose(np.linalg.inv(H) @ global_coord.T)
    # print(local_coord)

    local_data = np.hstack((local_coord[:, :2], global_data[:, 2].reshape(global_data.shape[0], 1),
                            global_data[:, 3].reshape(global_data.shape[0], 1)))  # stack v & theta
    # print(local_data)

    return local_data


def local_to_global(obs, local_data):
    x, y, theta = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
    H = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                  [np.sin(theta), np.cos(theta), 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # vehicle frame to world frame

    local_coord = np.hstack((local_data[:, :2], np.tile([0, 1], (local_data.shape[0], 1))))
    global_coord = np.transpose(H @ local_coord.T)
    # print(local_coord)

    global_data = np.hstack((global_coord[:, :2], local_data[:, 2].reshape(local_data.shape[0], 1),
                             local_data[:, 3].reshape(local_data.shape[0], 1)))  # stack v and theta
    # print(global_data)

    return global_data


def bresenham_line_point(p1, p2):
    points = []
    dx = abs(p2[0] - p1[0])
    dy = -abs(p2[1] - p1[1])
    sx = 0.1 if p1[0] < p2[0] else -0.1
    sy = 0.1 if p1[1] < p2[1] else -0.1
    # sx = 1 if p1[0] < p2[0] else -1
    # sy = 1 if p1[1] < p2[1] else -1

    err = dx + dy
    x0 = p1[0]
    y0 = p1[1]
    while True:
        points.append((x0, y0))
        # if x0 == p2[0] and y0 == p2[1]:
        if abs(x0 - p2[0]) < 0.2 and abs(y0 - p2[1]) < 0.2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return np.array(points)


def bresenham_line_index(p1, p2):
    points = []
    dx = abs(p2[0] - p1[0])
    dy = -abs(p2[1] - p1[1])
    # sx = 0.1 if p1[0] < p2[0] else -0.1
    # sy = 0.1 if p1[1] < p2[1] else -0.1
    sx = 1 if p1[0] < p2[0] else -1
    sy = 1 if p1[1] < p2[1] else -1

    err = dx + dy
    x0 = p1[0]
    y0 = p1[1]
    while True:
        points.append((x0, y0))
        if x0 == p2[0] and y0 == p2[1]:
            # if abs(x0 - p2[0])<0.2 and abs(y0 - p2[1])<0.2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return np.array(points)


def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


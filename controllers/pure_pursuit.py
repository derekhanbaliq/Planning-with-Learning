import numpy as np
from scipy.spatial import distance


class PurePursuit:
    """
    Implement Pure Pursuit on the car
    """

    def __init__(self, waypoints, L):
        self.is_clockwise = False

        self.waypoints = np.array([waypoints.x, waypoints.y]).T
        self.numWaypoints = self.waypoints.shape[0]
        self.ref_speed = waypoints.v

        self.L = L  # 1.5
        self.steering_gain = 0.5

    def control(self, obs, agent=1, offset=None):
        # Get current pose
        self.currX = obs['poses_x'][agent - 1]
        self.currY = obs['poses_y'][agent - 1]
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints,))
        self.closest_index = np.argmin(self.distances)

        # Find target point
        targetPoint, target_point_index = self.get_closest_point_beyond_lookahead_dist(self.L)

        if isinstance(offset, np.ndarray):  # agent == 1:
            targetPoint = targetPoint + offset  # += is not overwritten by np!

        # calculate steering angle / curvature
        waypoint_y = np.dot(np.array([np.sin(-obs['poses_theta'][agent - 1]), np.cos(-obs['poses_theta'][agent - 1])]),
                            targetPoint - np.array([self.currX, self.currY]))
        gamma = 2.0 * waypoint_y / self.L ** 2
        steering_angle = self.steering_gain * gamma
        # radius = 1 / (2.0 * waypoint_y / self.L ** 2)
        # steering_angle = np.arctan(0.33 / radius)  # Billy's method, but it also involves tricky fixing
        steering_angle = np.clip(steering_angle, -0.35, 0.35)

        # calculate speed
        speed = self.ref_speed[target_point_index]

        return steering_angle, speed

    def rl_control(self, obs, profile, max_speed=5.0, agent=1):
        target_point = np.array([profile[0], profile[1]])
        y = np.dot(np.array([np.sin(-obs['poses_theta'][agent - 1]), np.cos(-obs['poses_theta'][agent - 1])]),
                   target_point - np.array([obs['poses_x'][agent - 1], obs['poses_y'][agent - 1]]))
        gamma = 2.0 * y / self.L ** 2

        steering = self.steering_gain * gamma
        steering = np.clip(steering, -0.35, 0.35)

        speed = profile[2]
        speed = np.clip(speed, 0.0, max_speed)

        return steering, speed

    def get_closest_point_beyond_lookahead_dist(self, threshold):
        point_index = self.closest_index
        dist = self.distances[point_index]

        while dist < threshold:
            if self.is_clockwise:
                point_index -= 1
                if point_index < 0:
                    point_index = len(self.waypoints) - 1
                dist = self.distances[point_index]
            else:
                point_index += 1
                if point_index >= len(self.waypoints):
                    point_index = 0
                dist = self.distances[point_index]

        point = self.waypoints[point_index]

        return point, point_index

    def get_target_waypoint(self, obs, agent=1):
        # Get current pose
        self.currX = obs['poses_x'][agent - 1]
        self.currY = obs['poses_y'][agent - 1]
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints,))
        self.closest_index = np.argmin(self.distances)

        # Find target point
        targetPoint, target_point_index = self.get_closest_point_beyond_lookahead_dist(self.L)
        # print(f"agent num: {agent} at {targetPoint}")
        # self.targetPoint = targetPoint
        return targetPoint, target_point_index


def get_lookahead_point(obs, dense_traj, lookahead_dist=0.8):
    car_pose = np.array([obs['poses_x'][0], obs['poses_y'][0]])
    for i in range(dense_traj.shape[0] - 1):
        if np.linalg.norm(dense_traj[i][:2] - car_pose, ord=2) >= lookahead_dist:
            return dense_traj[i]
    return dense_traj[dense_traj.shape[0] - 1]


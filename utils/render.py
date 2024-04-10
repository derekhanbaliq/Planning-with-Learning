import numpy as np
from pyglet.gl import GL_POINTS  # game interface


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.drawn_waypoints = []
        self.obs = None

        self.traj = None
        self.last_traj = []

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints.x, self.waypoints.y)).T  # N x 2

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [255, 255, 255]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def load_obs(self, obs):
        self.obs = obs

    def render_path(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        x = self.obs['poses_x']
        y = self.obs['poses_y']

        point = np.array([x, y])

        scaled_point = 50. * point

        b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0.]),
                        ('c3B/stream', [255, 0, 0]))
        self.drawn_waypoints.append(b)

    def render_traj(self, e):
        """
        update reference trajectory
        """
        pass

        if self.traj.shape[1] == 4:
            x = self.traj[:, 1]
            y = self.traj[:, 2]
        else:
            x = self.traj[:, 0]
            y = self.traj[:, 1]

        point = np.vstack((x, y)).T

        scaled_point = 50. * point

        for last_point in self.last_traj:
            last_point.delete()
        self.last_traj.clear()

        for i in range(scaled_point.shape[0]):
            b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[i, 0], scaled_point[i, 1], 0.]),
                            ('c3B/stream', [255, 0, 0]))
            self.last_traj.append(b)

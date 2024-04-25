import numpy as np
# from pyglet.gl import *  # game interface
from pyglet.gl import GL_POINTS, GL_QUADS, GL_TRIANGLES  # game interface


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.drawn_waypoints = []

        self.front_traj = None
        self.last_front_traj = []

        self.horizon_traj = None
        self.last_horizon_traj = []

        self.ahead_point = None
        self.last_ahead_point = None

        self.offset_traj = None
        self.last_offset_traj = []

        self.lidar_data = None
        self.last_lidar_data = []

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

    def render_front_traj(self, e):
        """
        update reference front trajectory
        """

        if self.front_traj.shape[1] == 4:
            point = self.front_traj[:, 1:3]
        else:
            point = self.front_traj[:, :2]

        # point = np.vstack((x, y)).T
        scaled_point = 50. * point

        for last_point in self.last_front_traj:
            last_point.delete()
        self.last_front_traj.clear()

        for i in range(scaled_point.shape[0]):
            b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[i, 0], scaled_point[i, 1], 0.]),
                            ('c3B/stream', [255, 0, 0]))
            self.last_front_traj.append(b)

    def render_horizon_traj(self, e):
        """
        update horizon points
        """

        def create_triangle(x, y, size):
            return ('v2f/stream', [x + size, y,
                                   x - size / 2, y - size,
                                   x - size / 2, y + size])

        point = self.horizon_traj[:, :2]
        scaled_point = 50. * point

        for last_point in self.last_horizon_traj:
            last_point.delete()
        self.last_horizon_traj.clear()

        for i in range(scaled_point.shape[0]):
            b = e.batch.add(3, GL_TRIANGLES, None, create_triangle(scaled_point[i, 0], scaled_point[i, 1], 4),
                            ('c3B/stream', [255, 255, 0, 255, 255, 0, 255, 255, 0]))
            self.last_horizon_traj.append(b)

    def render_lookahead_point(self, e):
        """
        update lookahead point
        """

        def create_triangle(x, y, size):
            return ('v2f/stream', [x, y + size,
                                   x - size, y - size / 2,
                                   x + size, y - size / 2])

        scaled_point = 50. * self.ahead_point

        if self.last_ahead_point is not None:
            self.last_ahead_point.delete()

        b = e.batch.add(3, GL_TRIANGLES, None, create_triangle(scaled_point[0], scaled_point[1], 10),
                        ('c3B/stream', [0, 255, 255, 0, 255, 255, 0, 255, 255]))
        self.last_ahead_point = b

    def render_offset_traj(self, e):
        """Update offset trajectory being drawn."""
        def create_triangle(x, y, size):
            return ('v2f/stream', [x + size, y,
                                   x - size / 2, y - size,
                                   x - size / 2, y + size])

        point = self.offset_traj
        scaled_point = 50. * point

        for last_point in self.last_offset_traj:
            last_point.delete()
        self.last_offset_traj.clear()

        for i in range(scaled_point.shape[0]):
            b = e.batch.add(3, GL_TRIANGLES, None, create_triangle(scaled_point[i, 0], scaled_point[i, 1], 6),
                            ('c3B/stream', [255, 0, 0, 255, 0, 0, 255, 0, 0]))
            self.last_offset_traj.append(b)

    def render_lidar_data(self, e):
        """Render the occupancy grid as Green points."""
        if self.lidar_data == []:
            pass
        else:
            point = self.lidar_data
            scaled_point = 50 * point

            for last_point in self.last_lidar_data:
                last_point.delete()
            self.last_lidar_data.clear()

            for i in range(scaled_point.shape[0]):
                b = e.batch.add(1, GL_POINTS, None,
                                ('v3f/stream', [scaled_point[i, 0], scaled_point[i, 1], 0]),
                                ('c3B/stream', [0, 255, 0]))  # Green color
                self.last_lidar_data.append(b)


def fix_gui(e):
    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 1000
    e.right = right + 1000
    e.top = top + 800
    e.bottom = bottom - 800


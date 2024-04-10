import numpy as np
from pyglet.gl import GL_POINTS  # game interface


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.drawn_waypoints = []

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

    def render_traj(self, e):
        """
        update reference trajectory
        """

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

    def render_lookahead_point(self, e):
        """
        update lookahead point
        """

        pass


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

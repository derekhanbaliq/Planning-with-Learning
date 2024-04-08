"""
    LQR Steering Controller
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
"""

import numpy as np
import math
from utils.utils import calc_nearest_point, pi_2_pi


class Waypoint:
    def __init__(self, map_name, csv_data=None):
        if map_name == 'Spielberg' or map_name == 'MoscowRaceway' or map_name == 'Catalunya':
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3]  # coordinate matters!
            self.γ = csv_data[:, 4]
        elif map_name == 'example' or map_name == 'icra' or map_name == 'levine':
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3] + math.pi / 2  # coordinate matters!
            self.γ = csv_data[:, 4]


class CarState:
    def __init__(self, x=0.0, y=0.0, θ=0.0, v=0.0):
        self.x = x
        self.y = y
        self.θ = θ
        self.v = v


class LKVMState:
    """
    Linear Kinematic Vehicle Model's state space expression
    """
    def __init__(self, e_l=0.0, e_l_dot=0.0, e_θ=0.0, e_θ_dot=0.0):
        # 4 states
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.e_θ = e_θ
        self.e_θ_dot = e_θ_dot
        # log old states
        self.old_e_l = 0.0
        self.old_e_θ = 0.0

    def update(self, e_l, e_θ, dt):
        self.e_l = e_l
        self.e_l_dot = (e_l - self.old_e_l) / dt
        self.e_θ = e_θ
        self.e_θ_dot = (e_θ - self.old_e_θ) / dt

        x = np.vstack([self.e_l, self.e_l_dot, self.e_θ, self.e_θ_dot])

        return x


class PID:
    def __init__(self, Kp, Ki, Kd, i_limit=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.i_limit = i_limit
        self.pre_error = 0.0

    def output(self, error):
        self.error = error

        self.integral += self.Ki * self.error
        if self.integral > self.i_limit:
            self.integral = self.i_limit
        elif self.integral < -self.i_limit:
            self.integral = -self.i_limit

        self.derivative = self.error - self.pre_error

        output = self.Kp * self.error + self.integral + self.Kd * self.derivative

        self.pre_error = self.error

        return output


class LQR:
    def __init__(self, dt, wheelbase, v=0.0):
        self.A = np.array([[1.0, dt, 0, 0],
                           [0, 0, v, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 0]])
        self.B = np.array([[0],
                           [0],
                           [0],
                           [v / wheelbase]])
        # self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])  # Billy's code recommendation
        self.Q = np.diag([1, 0.0, 0.01, 0.0])
        self.R = np.diag([1])

    def discrete_lqr(self):
        A = self.A
        B = self.B
        R = self.R

        S = self.solve_recatti_equation()
        K = -np.linalg.pinv(B.T @ S @ B + R) @ (B.T @ S @ A)  # u = -(B.T @ S @ B + R)^(-1) @ (B.T @ S @ A) @ x[k]

        return K  # K is 4 x 1

    def solve_recatti_equation(self):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R  # just for simplifying the following recatti expression

        S = self.Q
        Sn = None

        max_iter = 100
        ε = 0.001  # tolerance epsilon
        diff = math.inf  # always use value iteration with max iteration!

        # print('S0 = Q = {}'.format(self.Q))

        i = 0
        while i < max_iter and diff > ε:
            i += 1
            Sn = Q + A.T @ S @ A - (A.T @ S @ B) @ np.linalg.pinv(R + B.T @ S @ B) @ (B.T @ S @ A)
            S = Sn

        # print('Sn = {}'.format(Sn))

        return Sn


class LQRSteeringController:

    def __init__(self, waypoints):
        self.dt = 0.01  # time step
        self.wheelbase = 0.33
        self.waypoints = waypoints
        self.car = CarState()
        self.x = LKVMState()  # whenever create the controller, x exists - relatively static
        self.pid = PID(0.1, 0.018, 0.01)

    def control(self, curr_obs):
        """
            input car_state & waypoints
            output lqr-steering & pid-speed
        """
        self.car.x = curr_obs['poses_x'][0]
        self.car.y = curr_obs['poses_y'][0]
        self.car.θ = curr_obs['poses_theta'][0]
        self.car.v = curr_obs['linear_vels_x'][0]  # each agent’s current longitudinal velocity

        steering = self.lqr_steering_control()
        speed = self.pid_speed_control()

        return steering, speed

    def lqr_steering_control(self):
        """
        LQR steering control for Lateral Kinematics Vehicle Model - only steering for this part, consider feedforward
        """

        self.x.old_e_l = self.x.e_l
        self.x.old_e_θ = self.x.e_θ  # log into x's static variables

        e_l, e_θ, γ, v = self.calc_control_points()  # Calculate errors and reference point

        lqr = LQR(self.dt, self.wheelbase, self.car.v)  # init A B Q R with the current car state
        K = lqr.discrete_lqr()  # use A, B, Q, R to get K

        x_new = self.x.update(e_l, e_θ, self.dt)  # x[k+1]

        feedback_term = (K @ x_new)[0, 0]  # K is 4 x 1 since u is 1 x 1, look out the signal of K!
        # feedforward_term = math.atan2(self.wheelbase * γ, 1)  # = math.atan2(L / r, 1) = math.atan2(L, r)
        feedforward_term = self.wheelbase * γ

        steering = - feedback_term + feedforward_term

        return steering

    def pid_speed_control(self):
        """
        use PID controller to control the speed
        """
        front_pos = self.get_front_pos()
        _, _, _, i = calc_nearest_point(front_pos, np.array([self.waypoints.x, self.waypoints.y]).T)

        error = self.waypoints.v[i] - self.car.v
        speed = self.pid.output(error)

        if speed >= 8.0:
            speed = 8.0  # speed limit < 8 m/s

        return speed

    def get_front_pos(self):
        front_x = self.car.x + self.wheelbase * math.cos(self.car.θ)
        front_y = self.car.y + self.wheelbase * math.sin(self.car.θ)
        front_pos = np.array([front_x, front_y])

        return front_pos

    def calc_control_points(self):
        front_pos = self.get_front_pos()

        waypoint_i, min_d, _, i = \
            calc_nearest_point(front_pos, np.array([self.waypoints.x, self.waypoints.y]).T)

        waypoint_to_front = front_pos - waypoint_i  # regard this as a vector

        front_axle_vec_rot_90 = np.array([[math.cos(self.car.θ - math.pi / 2.0)],
                                          [math.sin(self.car.θ - math.pi / 2.0)]])
        e_l = np.dot(waypoint_to_front.T, front_axle_vec_rot_90)  # real lateral error, the horizontal dist

        e_θ = pi_2_pi(self.waypoints.θ[i] - self.car.θ)  # heading error
        γ = self.waypoints.γ[i]  # curvature of the nearst waypoint
        v = self.waypoints.v[i]  # velocity of the nearst waypoint

        return e_l, e_θ, γ, v

    def get_error(self):
        e_l = self.x.e_l
        e_θ = self.x.e_θ

        return np.array([e_l, e_θ])

"""
    LQR Steering Speed Controller
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
"""

import numpy as np
import math
from controllers.lqr_steering import calc_nearest_point, pi_2_pi


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
    def __init__(self, e_l=0.0, e_l_dot=0.0, e_θ=0.0, e_θ_dot=0.0, v=0.0):
        # 4 states
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.e_θ = e_θ
        self.e_θ_dot = e_θ_dot
        self.e_v = v
        # log old states
        self.old_e_l = 0.0
        self.old_e_θ = 0.0

    def update(self, e_l, e_θ, e_v, dt):  # pack-up rather than calc
        self.e_l = e_l
        self.e_l_dot = (e_l - self.old_e_l) / dt
        self.e_θ = e_θ
        self.e_θ_dot = (e_θ - self.old_e_θ) / dt
        self.e_v = e_v  # e_l, e_θ, e_v have been calculated previously

        x = np.vstack([self.e_l, self.e_l_dot, self.e_θ, self.e_θ_dot, self.e_v])

        return x


class LQR:
    def __init__(self, dt, l_wb, v=0.0):
        self.A = np.array([[1.0,    dt,     0,       0,         0],
                           [0,      0,      v,       0,         0],
                           [0,      0,      1.0,     dt,        0],
                           [0,      0,      0,       0,         0],
                           [0,      0,      0,       0,         1.0]])
        self.B = np.array([[0,          0],
                           [0,          0],
                           [0,          0],
                           [v / l_wb,   0],
                           [0,          dt]])  # l_wb is wheelbase
        self.Q = np.diag([1, 0.0, 0.01, 0.0, 1])
        self.R = np.diag([1, 1])

    def discrete_lqr(self):
        A = self.A
        B = self.B
        R = self.R

        S = self.solve_recatti_equation()
        K = -np.linalg.pinv(B.T @ S @ B + R) @ (B.T @ S @ A)  # u = -(B.T @ S @ B + R)^(-1) @ (B.T @ S @ A) @ x[k]

        return K  # K is 2 x 5

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


class LQRSteeringSpeedController:
    def __init__(self, waypoints):
        self.dt = 0.01  # time step
        self.wheelbase = 0.33
        self.waypoints = waypoints
        self.car = CarState()
        self.x = LKVMState()  # whenever create the controller, x exists - relatively static

    def control(self, curr_obs):
        """
            input car_state & waypoints
            output lqr-steering & pid-speed
        """
        self.car.x = curr_obs['poses_x'][0]
        self.car.y = curr_obs['poses_y'][0]
        self.car.θ = curr_obs['poses_theta'][0]
        self.car.v = curr_obs['linear_vels_x'][0]  # each agent’s current longitudinal velocity

        steering, speed = self.lqr_steering_speed_control()

        return steering, speed

    def lqr_steering_speed_control(self):
        """
        LQR steering speed control for Lateral Kinematics Vehicle Model
        """

        self.x.old_e_l = self.x.e_l
        self.x.old_e_θ = self.x.e_θ  # log into x's static variables

        e_l, e_θ, γ, e_v = self.calc_control_points()  # Calculate errors and reference point

        lqr = LQR(self.dt, self.wheelbase, self.car.v)  # init A B Q R with the current car state
        K = lqr.discrete_lqr()  # use A, B, Q, R to get K

        x_new = self.x.update(e_l, e_θ, e_v, self.dt)  # x[k+1]

        steering_fb = (K @ x_new)[0, 0]  # K is 2 x 5, x is 5 x 1
        # feedforward_term = math.atan2(self.wheelbase * γ, 1)  # = math.atan2(L / r, 1) = math.atan2(L, r)
        steering_ff = self.wheelbase * γ  # can be seen as current steering angle

        speed_fb = (K @ x_new)[1, 0]  # the acceleration, should be regarded as Δv, or acceleration in 0.01s

        steering = - steering_fb + steering_ff
        speed = - speed_fb + self.car.v  # current car speed is the base, v_base + Δv is the speed we want
        # speed = - accel * self.dt + self.car.v is wrong because we are in the loop, Δt should be "unit 1"

        if speed >= 8.0:
            speed = 8.0  # speed limit < 8 m/s

        return steering, speed

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

        e_v = self.waypoints.v[i] - self.car.v  # velocity of the nearst waypoint

        return e_l, e_θ, γ, e_v

    def get_error(self):
        e_l = self.x.e_l
        e_θ = self.x.e_θ

        return np.array([e_l, e_θ])

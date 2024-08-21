"""
    Single Track Kinematic MPC - STKMPC
    Author: Hongrui Zheng, Johannes Betz, Ahmad Amine, Derek Zhou
    References: https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/kinematic_mpc
                https://github.com/f1tenth/f1tenth_planning/tree/main/examples/control
                https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
                https://www.cvxpy.org/
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control
"""
from dataclasses import dataclass, field

import cvxpy
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix


@dataclass
class MPCConfig_F110:
    NXK: int = 4
    NU: int = 2
    TK: int = 10  # rising to 12 will cut corners and unstable straight running

    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 35.0])  # 25 will cause unstable straight running
    )  # input cost matrix, penalty for inputs - [accel, steering_angle]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 35.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_angle]
    Qk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 20])
    )  # state error cost matrix, for the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 20])
    )  # final state error matrix, penalty for the final state constraints: [x, y, v, yaw]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    # dlk: float = 0.03  # dist step [m] kinematic
    dlk: float = 0.2  # dist step [m] kinematic - check the difference between waypoints[0, 0] and waypoints[0, 1]
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    LR: float = 0.17145
    LF: float = 0.15875
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad], 24.00°
    # MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s], 3.141592653589793
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 2.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MAX_DECEL: float = -3.0  # maximum acceleration [m/ss]
    MASS: float = 3.74  # Vehicle mass


class KinematicModel:
    """
    states - [x, y, v, yaw]
    inputs - [acceleration, steering angle]
    reference point - center of rear axle
    """

    def __init__(self, config):
        self.config = config

    def clip_input(self, u):
        # u matrix N x 2
        u = np.clip(u, [self.config.MAX_DECEL, self.config.MIN_STEER], [self.config.MAX_ACCEL, self.config.MAX_STEER])
        # numpy.clip(a, a_min, a_max, out=None, **kwargs), Clip (limit) the values in an array.

        return u

    def clip_output(self, state):
        # state matrix N x 4
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)  # speed only

        return state

    def get_model_constraints(self):
        state_constraints = np.array([[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf],
                                      [np.inf, np.inf, self.config.MAX_SPEED, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])
        input_diff_constraints = np.array([[-np.inf, -self.config.MAX_STEER_V * self.config.DTK],
                                           [np.inf, self.config.MAX_STEER_V * self.config.DTK]])

        return state_constraints, input_constraints, input_diff_constraints

    def sort_reference_trajectory(self, position_ref, yaw_ref, speed_ref):
        reference = np.array([position_ref[:, 0], position_ref[:, 1], speed_ref, yaw_ref])  # x, y, v, yaw

        return reference  # N x 4

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]

        return speed, orientation, position  # express the states more generally

    def get_f(self, state, control_input):
        # state = x, y, v, yaw
        clipped_control_input = self.clip_input(control_input)  # input check
        delta = clipped_control_input[1]
        a = clipped_control_input[0]

        # f is for Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
        f = np.zeros(4)
        f[0] = state[2] * np.cos(state[3])  # x_dot
        f[1] = state[2] * np.sin(state[3])  # y_dot
        f[3] = state[2] / self.config.WB * np.tan(delta)  # yaw_dot
        f[2] = a  # v_dot

        return f  # kinematic model f(x[k], u[k]), Automatic Steering P27 or Atsushi's KMPC doc

    def get_model_matrix(self, state, u):
        """
        https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
        Calculate kinematic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]
        """
        v = state[2]
        phi = state[3]
        delta = u[1]

        # State (or system) matrix A, 4 x 4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * np.cos(phi)
        A[0, 3] = -self.config.DTK * v * np.sin(phi)
        A[1, 2] = self.config.DTK * np.sin(phi)
        A[1, 3] = self.config.DTK * v * np.cos(phi)
        A[3, 2] = self.config.DTK * np.tan(delta) / self.config.WB

        # Input Matrix B, 4 x 2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * np.cos(delta) ** 2)

        # Matrix C, 4 x 1, C is just a shift because we need an affine model
        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * np.sin(phi) * phi
        C[1] = -self.config.DTK * v * np.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * np.cos(delta) ** 2)

        return A, B, C

    def predict_motion(self, x0, control_input):
        # derive the future steps by Forward Euler Discretization - predict
        predicted_states = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1)
        predicted_states[:, 0] = x0  # set current state
        state = x0
        for i in range(1, self.config.TK + 1):  # 1 ... 8
            # Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
            state = state + self.get_f(state, control_input[:, i - 1]) * self.config.DTK
            state = self.clip_output(state)
            predicted_states[:, i] = state

        input_prediction = np.zeros((self.config.NU, self.config.TK + 1))  # 2 x (8 + 1), empty!

        return predicted_states, input_prediction  # filled states, empty inputs


class KMPCController:
    """
    Single Track Kinematic MPC Controller
    waypoints are just whole CSV data
    """

    def __init__(self, model, config, waypoints=None):
        self.waypoints = waypoints
        self.model = model
        self.config = config
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.input_o = np.zeros(self.config.NU) * np.NAN

        # placeholders for self.vars in mpc_prob_init()
        self.xk = None
        self.uk = None
        self.x0k = None
        self.ref_traj_k = None
        self.Annz_k = None
        self.Ak_ = None
        self.Bnnz_k = None
        self.Bk_ = None
        self.Ck_ = None
        self.MPC_prob = None
        self.mpc_prob_init()

    def control(self, states, waypoints=None):
        """
        input waypoints and current car states, execute MPC to return steering and speed with other data for logging
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
            self.waypoints = waypoints
            self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        else:
            if self.waypoints is None:
                raise ValueError("Please set waypoints to track during planner instantiation or when calling control()")

        steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy, a = self.MPC_Control(
            states, self.waypoints)

        return steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy, a

    def rl_control(self, states, offset_traj):
        x0 = states
        path = self.waypoints

        # get current state for calculating reference trajectory
        # speed, orientation, position = self.model.get_general_states(x0)  # v, yaw, [x, y]
        # interpolated waypoints for ref traj - can be a great visualization tool for debugging waypoint calculation
        # ref_path = self.calc_ref_trajectory(position, orientation, speed, path)
        ref_yaw = offset_traj[:, -1]
        ref_yaw[ref_yaw - states[3] > 5.0] = np.abs(
            ref_yaw[ref_yaw - states[3] > 5.0] - (2 * np.pi)
        )
        ref_yaw[ref_yaw - states[3] < -5.0] = np.abs(
            ref_yaw[ref_yaw - states[3] < -5.0] + (2 * np.pi)
        )
        refined_offset_traj = np.hstack((offset_traj[:, :3], ref_yaw.reshape((-1, 1))))
        # print(refined_offset_traj.shape)
        self.rlmpc_ref_path = refined_offset_traj.T

        # Solve the Linear MPC Control problem
        self.input_o, states_output, state_predict = self.linear_mpc_control(self.rlmpc_ref_path, x0, self.input_o)

        # Steering Output: First entry of the MPC steering angle output vector in degree
        u = self.input_o[:, 0]
        steering = u[1]
        speed = u[0] * self.config.DTK + x0[2]  # speed must add the base speed ~ v = v0 + a * dt

        ox = states_output[0]
        oy = states_output[1]  # a series of solved x & y

        # solved steering & speed for next step, ref / predicted / solved series of x & y, u[0] for ext-KMPC
        return steering, speed, self.rlmpc_ref_path[0], self.rlmpc_ref_path[1], state_predict[0], state_predict[1], ox, oy, u[0]
        # return steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy, a

    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx, waypoints):
        s_relative = np.zeros((self.config.TK + 1,))  # horizon + 1
        s_relative[0] = dist_from_segment_start  # nearest dist
        s_relative[1:] = predicted_speeds * self.config.DTK  # curr speed * time step (0.1s)
        s_relative = np.cumsum(s_relative)  # accumulated distance in horizon time steps based on nearest dist
        # cumsum(): Return the cumulative sum of the elements along a given axis.

        waypoints_distances_relative = np.cumsum(np.roll(self.waypoints_distances, -idx))
        # roll(): Roll array elements along a given axis. -idx = roll in left dir for idx step

        index_relative = np.int_(np.ones((self.config.TK + 1,)))  # record increasing idx based on curr idx
        for i in range(self.config.TK + 1):  # horizon num of times
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()  # idx[i] = [T T F F...].sum() = 2
        index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)  # avoid index overflow
        # print("index_absolute = {}".format(index_absolute))

        # remainder after subtracting wpt step dist (0.2)
        segment_part = \
            s_relative - (waypoints_distances_relative[index_relative] - self.waypoints_distances[index_absolute])

        t = (segment_part / self.waypoints_distances[index_absolute])  # segment's ratio
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                          waypoints[index_absolute][:, (1, 2)])  # (horizon + 1) * 2
        orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                             waypoints[index_absolute][:, 3])
        speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                       waypoints[index_absolute][:, 5])

        interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
        interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
        interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)

        # rearrange reference data
        reference = self.model.sort_reference_trajectory(interpolated_positions,
                                                         interpolated_orientations,
                                                         interpolated_speeds)  # N x 4 data

        return reference

    def calc_ref_trajectory(self, position, orientation, speed, path):
        """
        https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control
        calc reference trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        """
        # Find the nearest index & dist current pos
        _, dist, _, _, ind = nearest_point(np.array([position[0], position[1]]), path[:, (1, 2)])

        # get interpolated waypoints for reference
        reference = self.get_reference_trajectory(np.ones(self.config.TK) * abs(speed), dist, ind, path)  # 4 x (h + 1)

        # TODO: to be improved
        # check the yaw angle difference is over 5 or not, to avoid the abrupt 2π change (≈ 2π but < 2π)
        reference[3, :][reference[3, :] - orientation > 5] = \
            np.abs(reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = \
            np.abs(reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        # normal pi_2_pi works as well, but according to Ahmad, threshold-5 is smoother
        # reference[3, :][reference[3, :] - orientation > 5] = \
        #     pi_2_pi(reference[3, :][reference[3, :] - orientation > 5])

        return reference

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Problem will be solved for every control iteration.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        """
        # Initialize and create vectors for the optimization problem
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))  # Vehicle State Vector, 4 x (8 + 1)
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))  # Control Input vector, 2 x 8
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))  # 4 x 1
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1) ~ Qk + Qfk
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))  # (2 x 2) * 8, diagonal matrix

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))  # (2 x 2) * (8 - 1) ~ 7, difference

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * self.config.TK  # (4 x 4) * 8
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))  # (4 x 4) * (8 + 1)

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        # cvxpy.vec() - Flattens the matrix X into a vector in column-major order

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep
        #              T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)  # Qk + Qfk

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []

        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1)
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))  # 2 x (8 + 1)
        for t in range(self.config.TK):  # 8
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))  # A_block changes from list to <class 'scipy.sparse._coo.coo_matrix'>
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)  # 32 x 1

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape  # 32, 32
        self.Annz_k = cvxpy.Parameter(A_block.nnz)  # nnz: number of nonzero elements, nnz = 128
        data = np.ones(self.Annz_k.size)  # 128 x 1, size = 128, all elements are 1
        rows = A_block.row * n + A_block.col  # No. ? element in 32 x 32 matrix
        cols = np.arange(self.Annz_k.size)  # 128 elements that need to be care - diagonal & nonzero, 4 x 4 x 8
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))  # (rows, cols)	data

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data  # real data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")
        # https://www.cvxpy.org/api_reference/cvxpy.atoms.affine.html#cvxpy.reshape

        # B, Same as A
        m, n = B_block.shape  # 32, 16 = 4 x 8, 2 x 8
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)  # nnz = 64
        data = np.ones(self.Bnnz_k.size)  # 64 = (4 x 2) x 8
        rows = B_block.row * n + B_block.col  # No. ? element in 32 x 16 matrix
        cols = np.arange(self.Bnnz_k.size)  # 0, 1, ... 63
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))  # (rows, cols)	data

        # sparse version instead of the old B_block
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")

        # real data
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # Add dynamics constraints to the optimization problem
        constraints += [cvxpy.vec(self.xk[:, 1:])
                        == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_]
        # cvxpy.vec() - Flattens the matrix X into a vector in column-major order

        # Constraint 2: initial state - set x[k=0] as x0
        constraints += [self.xk[:, 0] == self.x0k]

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        state_constraints, input_constraints, input_diff_constraints = self.model.get_model_constraints()

        for i in range(self.config.NXK):  # Constraint 3: state constraints
            constraints += [state_constraints[0, i] <= self.xk[i, :], self.xk[i, :] <= state_constraints[1, i]]

        for i in range(self.config.NU):  # Constraint 4: input constraints
            constraints += [input_constraints[0, i] <= self.uk[i, :], self.uk[i, :] <= input_constraints[1, i]]
            constraints += [input_diff_constraints[0, i] <= cvxpy.diff(self.uk[i, :]),
                            cvxpy.diff(self.uk[i, :]) <= input_diff_constraints[1, i]]

        # Create the optimization problem in CVXPY and setup the workspace
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)  # minimize the objective function

    def mpc_prob_solve(self, ref_traj, path_predict, x0, input_predict):
        # init state value
        self.x0k.value = x0

        # get A, B, C block matrices
        A_block = []
        B_block = []
        C_block = []

        for t in range(self.config.TK):
            # use predicted data to optimize the objective function to min, but only use first x & u for execution
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.01, eps_abs=0.0005,
                            eps_rel=0.0005, verbose=False, warm_start=True)
        # verbose shows the log, other params limit the accuracy and iterations
        # we don't need the extreme precision to 10e-6 with the tolerance of 10000 iterations

        # solves the problem with desired accuracy or have a lower accuracy than desired
        if self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE:
            o_states = self.xk.value
            ou = self.uk.value
        else:
            print("Error: Cannot solve KS mpc... Status : ", self.MPC_prob.status)
            ou, o_states = np.zeros(self.config.NU) * np.NAN, np.zeros(self.config.NXK) * np.NAN  # clear

        return ou, o_states  # data with a horizon length of optimal states and inputs

    def linear_mpc_control(self, ref_path, x0, ref_control_input):
        """
        MPC control with updating operational point iteratively
        """
        # clear input
        if np.isnan(ref_control_input).any():
            ref_control_input = np.zeros((2, self.config.TK))

        # Predict the vehicle motion for num-of-horizon steps using Forward Euler Discretization
        state_prediction, input_prediction = self.model.predict_motion(x0, ref_control_input)
        # set any input prediction to become zero

        # Run the MPC optimization: solve the optimization problem
        mpc_input_output, mpc_states_output = self.mpc_prob_solve(ref_path, state_prediction, x0, input_prediction)
        # optimal inputs & states with in coming horizon steps

        return mpc_input_output, mpc_states_output, state_prediction

    def MPC_Control(self, x0, path):  # input current vehicle state & waypoints (== path)
        # get current state for calculating reference trajectory
        speed, orientation, position = self.model.get_general_states(x0)  # v, yaw, [x, y]
        # interpolated waypoints for ref traj - can be a great visualization tool for debugging waypoint calculation
        ref_path = self.calc_ref_trajectory(position, orientation, speed, path)

        # Solve the Linear MPC Control problem
        self.input_o, states_output, state_predict = self.linear_mpc_control(ref_path, x0, self.input_o)

        # Steering Output: First entry of the MPC steering angle output vector in degree
        u = self.input_o[:, 0]
        steering = u[1]
        speed = u[0] * self.config.DTK + x0[2]  # speed must add the base speed ~ v = v0 + a * dt

        ox = states_output[0]
        oy = states_output[1]  # a series of solved x & y

        # solved steering & speed for next step, ref / predicted / solved series of x & y, u[0] for ext-KMPC
        return steering, speed, ref_path[0], ref_path[1], state_predict[0], state_predict[1], ox, oy, u[0]


def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the env
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points
                   on the trajectory. (p_i---*-------p_i+1)
        i (int): index of the nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])

    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


class KinematicModel:
    """
    states - [x, y, v, yaw]
    inputs - [acceleration, steering angle]
    reference point - center of rear axle
    """

    def __init__(self, config):
        self.config = config

    def clip_input(self, u):
        # u matrix N x 2
        u = np.clip(u, [self.config.MAX_DECEL, self.config.MIN_STEER], [self.config.MAX_ACCEL, self.config.MAX_STEER])
        # numpy.clip(a, a_min, a_max, out=None, **kwargs), Clip (limit) the values in an array.

        return u

    def clip_output(self, state):
        # state matrix N x 4
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)  # speed only

        return state

    def get_model_constraints(self):
        state_constraints = np.array([[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf],
                                      [np.inf, np.inf, self.config.MAX_SPEED, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])
        input_diff_constraints = np.array([[-np.inf, -self.config.MAX_STEER_V * self.config.DTK],
                                           [np.inf, self.config.MAX_STEER_V * self.config.DTK]])

        return state_constraints, input_constraints, input_diff_constraints

    def sort_reference_trajectory(self, position_ref, yaw_ref, speed_ref):
        reference = np.array([position_ref[:, 0], position_ref[:, 1], speed_ref, yaw_ref])  # x, y, v, yaw

        return reference  # N x 4

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]

        return speed, orientation, position  # express the states more generally

    def get_f(self, state, control_input):
        # state = x, y, v, yaw
        clipped_control_input = self.clip_input(control_input)  # input check
        delta = clipped_control_input[1]
        a = clipped_control_input[0]

        # f is for Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
        f = np.zeros(4)
        f[0] = state[2] * np.cos(state[3])  # x_dot
        f[1] = state[2] * np.sin(state[3])  # y_dot
        f[3] = state[2] / self.config.WB * np.tan(delta)  # yaw_dot
        f[2] = a  # v_dot

        return f  # kinematic model f(x[k], u[k]), Automatic Steering P27 or Atsushi's KMPC doc

    def get_model_matrix(self, state, u):
        """
        https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
        Calculate kinematic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]
        """
        v = state[2]
        phi = state[3]
        delta = u[1]

        # State (or system) matrix A, 4 x 4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * np.cos(phi)
        A[0, 3] = -self.config.DTK * v * np.sin(phi)
        A[1, 2] = self.config.DTK * np.sin(phi)
        A[1, 3] = self.config.DTK * v * np.cos(phi)
        A[3, 2] = self.config.DTK * np.tan(delta) / self.config.WB

        # Input Matrix B, 4 x 2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * np.cos(delta) ** 2)

        # Matrix C, 4 x 1, C is just a shift because we need an affine model
        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * np.sin(phi) * phi
        C[1] = -self.config.DTK * v * np.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * np.cos(delta) ** 2)

        return A, B, C

    def predict_motion(self, x0, control_input):
        # derive the future steps by Forward Euler Discretization - predict
        predicted_states = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1)
        predicted_states[:, 0] = x0  # set current state
        state = x0
        for i in range(1, self.config.TK + 1):  # 1 ... 8
            # Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
            state = state + self.get_f(state, control_input[:, i - 1]) * self.config.DTK
            state = self.clip_output(state)
            predicted_states[:, i] = state

        input_prediction = np.zeros((self.config.NU, self.config.TK + 1))  # 2 x (8 + 1), empty!

        return predicted_states, input_prediction  # filled states, empty inputs

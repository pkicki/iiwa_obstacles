import os
import os.path
import pickle
from copy import copy

from src.iiwa_obstacles.iiwa_obstacles_neural_planner.scripts.manifold_planning.utils.constants import Robot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import rosbag
from glob import glob

import pinocchio as pino
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot

from manifold_planning.utils.manipulator import Iiwa
from manifold_planning.utils.feasibility import check_if_plan_valid, compute_cartesian_losses

from manifold_planning.utils.constants import Table1, Table2, Cup, Limits

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

robot_file = os.path.join(root_dir, "manifold_planning", "iiwa_cup.urdf")
pino_model = pino.buildModelFromUrdf(robot_file)
pino_data = pino_model.createData()
joint_id = pino_model.getFrameId("F_link_cup")

package_path = os.path.join(os.path.dirname(__file__), "..")
file_name = os.path.join(package_path, "data/iiwa_obstacles_fairlysimple4/test/data.tsv")
data = np.loadtxt(file_name, delimiter="\t").astype(np.float32)  # [:3]

#q_dot_limits = 1.0 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=np.float32)[np.newaxis]
q_dot_limits = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=np.float32)[np.newaxis]
q_ddot_limits = 10. * q_dot_limits


def cal_d(fs, dts):
    dfs = np.zeros_like(fs)
    dfs[1: -1] = (fs[2:, :] - fs[:-2, :]) / (dts[2:] - dts[:-2])
    dfs[0] = dfs[1]
    dfs[-1] = dfs[-2]
    return dfs


def get_vel_acc(t, q_m):
    # Position
    bv, av = butter(6, 40, fs=1000)
    b, a = butter(6, 40, fs=1000)
    bt, at = butter(6, 30, fs=1000)
    # bv, av = butter(6, 4, fs=1000)
    # b, a = butter(6, 4, fs=1000)
    q_m_filter = q_m

    # Velocity
    dq_m = cal_d(q_m, t)
    dq_m_filter = filtfilt(bv, av, dq_m.copy(), axis=0)

    # Acceleration
    ddq_m = cal_d(dq_m_filter, t)
    ddq_m_filter = filtfilt(b, a, ddq_m.copy(), axis=0)

    return q_m_filter, dq_m_filter, ddq_m_filter, q_m, dq_m, ddq_m


def compute_vel_acc_tau(t, q, qd, qd_dot, qd_ddot):
    q, dq, ddq, q_, dq_, ddq_, = get_vel_acc(t[:, np.newaxis], q)
    return q, dq, ddq, qd, qd_dot, qd_ddot


def forwardKinematics(q):
    xyz = []
    R = []
    for i in range(len(q)):
        pino.forwardKinematics(pino_model, pino_data, q[i])
        pino.updateFramePlacements(pino_model, pino_data)
        xyz_pino = [x.translation for x in pino_data.oMf]
        xyz.append(copy(xyz_pino))
    xyz = np.stack(xyz)
    return xyz


def if_collision_free(q, obstacles):
    allowed_violation = 0.01
    xyz = forwardKinematics(q)

    dists = np.linalg.norm(xyz[:, np.newaxis] - obstacles[np.newaxis, :, np.newaxis, :3], axis=-1)
    radiuses = obstacles[np.newaxis, :, np.newaxis, -1] + Robot.radius - allowed_violation
    if np.any(dists < radiuses):
        return False

    return True


def read_bag(bag):
    time = []
    joint_state_desired = []
    joint_state_actual = []
    joint_state_error = []
    planning_time = None
    success = None
    planned_motion_time = None

    for topic, msg, t_bag in bag.read_messages():
        if topic in ["/bspline_ff_kino_joint_trajectory_controller/state"]:
            n_joints = len(msg.joint_names)
            time.append(msg.header.stamp.to_sec())
            joint_state_desired.append(np.concatenate(
                [msg.desired.positions, msg.desired.velocities, msg.desired.accelerations, msg.desired.effort]))
            joint_state_actual.append(np.concatenate(
                [msg.actual.positions, msg.actual.velocities, msg.actual.accelerations, msg.actual.effort]))
            joint_state_error.append(np.concatenate(
                [msg.error.positions, msg.error.velocities, msg.error.accelerations, msg.error.effort]))
        elif topic == "/neural_planner/status":
            planning_time = msg.planning_time
            planned_motion_time = msg.planned_motion_time
            success = msg.success
    desired = np.array(joint_state_desired).astype(np.float32)
    actual = np.array(joint_state_actual).astype(np.float32)
    error = np.array(joint_state_error).astype(np.float32)
    t = np.array(time)
    t -= t[0]
    t = t.astype(np.float32)
    return dict(t=t, actual=actual, desired=desired, error=error, planning_time=planning_time, success=success,
                planned_motion_time=planned_motion_time)


def compute_metrics(bag_path):
    # bag_path = os.path.join(package_dir, "bags", bag_path)
    # bag_path = os.path.join(package_dir, "bags/ours_nn/K9.bag")
    bag_file = rosbag.Bag(bag_path)
    i = int(bag_path.split("/")[-1][:-4])
    data_i = data[i]
    obstalces = np.reshape(data_i[14:], (-1, 4))

    result = {}

    bag_dict = read_bag(bag_file)
    q, dq, ddq, qd, qd_dot, qd_ddot = compute_vel_acc_tau(bag_dict["t"],
                                                          bag_dict["actual"][:, :7],
                                                          bag_dict["desired"][:, :7],
                                                          bag_dict["desired"][:, 7:14],
                                                          bag_dict["desired"][:, 14:21])
    for i in range(6):
        plt.subplot(321+i)
        plt.plot(bag_dict["t"], qd_dot[..., i], 'r')
        plt.plot(bag_dict["t"], dq[..., i], 'm')
        plt.plot([0, bag_dict["t"][-1]], [q_dot_limits[0, i], q_dot_limits[0, i]])
        plt.plot([0, bag_dict["t"][-1]], [-q_dot_limits[0, i], -q_dot_limits[0, i]])
    plt.show()
    moving = np.linalg.norm(qd_dot, axis=-1) > 5e-2
    if not np.any(moving):
        result["valid"] = 0
        result["finished"] = 0
        result["planning_time"] = bag_dict["planning_time"]
        return result
    q = q[moving]
    dq = dq[moving]
    ddq = ddq[moving]
    qd = qd[moving]
    qd_dot = qd_dot[moving]
    qd_ddot = qd_ddot[moving]
    t = bag_dict["t"][moving]

    # plt.plot(t, qd_dot[..., :2], 'b')
    # plt.show()

    ball_constraints = if_collision_free(q, obstalces)

    dq_constraints = np.all(np.abs(dq) < q_dot_limits)
    ddq_constraints = np.all(np.abs(ddq) < q_ddot_limits)
    # for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(dq[:, i]))
    #    plt.plot([0, dq.shape[0]], [q_dot_limits[0, i], q_dot_limits[0, i]])
    # plt.show()
    # for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(ddq[:, i]))
    #    plt.plot([0, ddq.shape[0]], [q_ddot_limits[0, i], q_ddot_limits[0, i]])
    # plt.show()
    # for i in range(7):
    #    plt.subplot(331 + i)
    #    plt.plot(np.abs(torque[:, i]))
    #    plt.plot([0, torque.shape[0]], [torque_limits[0, i], torque_limits[0, i]])
    # plt.show()

    dist_2_goal = np.linalg.norm(data_i[7:14] - q[-1])
    finished = dist_2_goal < 0.2

    # valid = box_constraints and vertical_constraints and torque_constraints and dq_constraints and ddq_constraints and finished
    # valid = box_constraints and vertical_constraints and torque_constraints and dq_constraints and finished
    valid = ball_constraints and finished

    #ee, _ = forwardKinematics(q)
    #ee_d, _ = forwardKinematics(qd)

    # plt.plot(ee[..., 0], ee[..., 1], 'r')
    # plt.plot(ee_d[..., 0], ee_d[..., 1], 'g')
    # plt.plot(bag_dict['puck_pose'][:, 0], bag_dict['puck_pose'][:, 1], 'b')
    # plt.show()

    # for i in range(6):
    #   plt.subplot(321 + i)
    #   plt.plot(t, q[..., i], 'r')
    #   plt.plot(t, qd[..., i], 'g')
    # plt.show()

    # for i in range(6):
    #    plt.subplot(321 + i)
    #    plt.plot(t, np.abs(q[..., i] - qd[..., i]))
    #    plt.plot(t, np.linalg.norm(ee_d[..., :2] - ee[..., :2], axis=-1))
    # plt.show()

    movement_time = t[-1] - t[0]

    dt = np.diff(t)
    integral = lambda x: np.sum(np.abs(x)[1:] * dt)
    # reduce_integral = lambda x: integral(np.sum(np.abs(x), axis=-1))
    reduce_integral = lambda x: integral(np.linalg.norm(x, axis=-1))

    result["valid"] = int(valid)
    result["finished"] = int(finished)
    result["planning_time"] = bag_dict["planning_time"]
    result["motion_time"] = movement_time

    result["joint_trajectory_error"] = reduce_integral(qd - q)
    #result["cartesian_trajectory_error"] = reduce_integral(ee_d - ee)

    print(result)
    return result


#planners = ["ours", "bitstar_0s", "bitstar_1s", "bitstar_10s"]
#planners = ["bitstar_10s"]
#planners = ["abitstar_0s", "aitstar_0s"]
# planners = ["nlopt", "sst", "cbirrt", "mpcmpnet"]
planners = ["ours"]
# planners = ["ours_long"]
# planners = ["cbirrt"]
# planners = ["nlopt"]
# planners = ["mpcmpnet"]
# planners = ["sst"]
package_dir = "/home/piotr/b8/ah_ws/data"
for planner in planners:
    print(planner)
    dir_path = os.path.join(package_dir, "obs_exp", planner)
    sp = dir_path.replace("data", "results")
    os.makedirs(sp, exist_ok=True)
    for i, p in enumerate(sorted(glob(os.path.join(dir_path, "*.bag")))):
        # for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [16, 23, 24, 42, 44, 67, 72]]):
        # for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [1, 12, 16, 24, 39, 42, 44, 65, 74, 91]]):
        # for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [0, 58]]):
        # for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [24, 44, 53, 80, 93, 94]]):
        # for i, p in enumerate(glob(os.path.join(dir_path, "00*.bag"))):
        # for i, p in enumerate(glob(os.path.join(dir_path, "*42.bag"))):
        # for i, p in enumerate([os.path.join(dir_path, f"{str(x).zfill(3)}.bag") for x in [0, 6, 16, 21, 23, 24, 42, 44, 65, 66, 67, 72, 92]]):
        print(i)
        d = compute_metrics(p)
        save_path = p[:-3] + "res"
        save_path = save_path.replace("data", "results")
        with open(save_path, 'wb') as fh:
            pickle.dump(d, fh)

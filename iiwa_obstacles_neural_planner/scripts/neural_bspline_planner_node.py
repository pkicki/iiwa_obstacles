#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import os
import sys
import inspect
import numpy as np
from copy import copy
import pinocchio as pino
from time import perf_counter
from scipy.interpolate import interp1d

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SCRIPT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
PLANNING_MODULE_DIR = os.path.join(SCRIPT_DIR, "manifold_planning")
sys.path.append(PLANNING_MODULE_DIR)
from iiwa_obstacles_msgs.msg import PlannerRequest, PlannerStatus
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Transform, Vector3
from iiwas_control.msg import BsplineTrajectoryMsg, BsplineSegmentMsg
from planner_request_utils import unpack_planner_request
from manifold_planning.utils.bspline import BSpline
from manifold_planning.utils.constants import UrdfModels, Limits, Base
from manifold_planning.utils.model import model_inference, load_model_obstacles, model_fast_inference


class NeuralPlannerNode:
    def __init__(self):
        rospy.init_node("neural_planner_node", anonymous=True)
        controller_type = rospy.get_param("~controllers", "bspline_ff_kino_joint_trajectory_controller")
        print(controller_type)
        planner_path = os.path.join(PACKAGE_DIR, rospy.get_param("/neural_planner/planner_path"))
        self.planning_request_subscriber = rospy.Subscriber("/neural_planner/plan_trajectory", PlannerRequest,
                                                            self.compute_trajectory)
        self.iiwa_publisher = rospy.Publisher(f"/{controller_type}/bspline",
                                              BsplineTrajectoryMsg,
                                              queue_size=5)
        self.cartesian_front_publisher = rospy.Publisher(f"/cartesian_trajectory", Path,
                                                         queue_size=5)
        self.planner_status_publisher = rospy.Publisher(f"/neural_planner/status", PlannerStatus, queue_size=5)
        self.urdf_path = os.path.join(PLANNING_MODULE_DIR, UrdfModels.iiwa)

        self.dim_q_control_points = 7
        self.num_q_control_points = 15
        self.num_t_control_points = 20
        self.bsp = BSpline(self.num_q_control_points, num_T_pts=64)
        self.bspt = BSpline(self.num_t_control_points, num_T_pts=64)
        print("Bspline initialized")
        print(planner_path)
        self.planner_model = load_model_obstacles(planner_path, self.num_q_control_points, self.bsp, self.bspt)
        print("planner model loaded")
        self.pino_model = pino.buildModelFromUrdf(self.urdf_path)
        self.pino_data = self.pino_model.createData()
        print("pino model loaded")
        print("node loaded")

    def compute_trajectory(self, msg):
        print("COMPUTE TRAJ")
        q_0, q_d, obstacles = unpack_planner_request(msg)

        d = np.concatenate([q_0, q_d, obstacles], axis=-1)[np.newaxis]
        d = d.astype(np.float32)
        t0 = perf_counter()
        q, dq, ddq, t, q_cps, t_cps = model_fast_inference(self.planner_model, d, self.bsp, self.bspt)
        #q_, dq_, ddq_, t_, q_cps_, t_cps_ = model_inference(self.planner_model, d, self.bsp, self.bspt)
        t1 = perf_counter()
        self.publish_joint_trajectory(q_cps, t_cps)
        print("PLANNING TIME: ", t1 - t0)
        self.publish_cartesian_trajectory(q, t)
        self.publish_planner_status(t1 - t0, t)

    def publish_joint_trajectory(self, qs, ts):
        iiwa_bspline = BsplineTrajectoryMsg()
        iiwa_bspline_segment = BsplineSegmentMsg()
        iiwa_bspline_segment.q_control_points = qs.flatten()
        iiwa_bspline_segment.t_control_points = ts.flatten()
        iiwa_bspline_segment.dim_q_control_points = self.dim_q_control_points
        iiwa_bspline_segment.num_q_control_points = self.num_q_control_points
        iiwa_bspline_segment.num_t_control_points = self.num_t_control_points
        iiwa_bspline.segments.append(iiwa_bspline_segment)
        iiwa_bspline.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        self.iiwa_publisher.publish(iiwa_bspline)

    def publish_cartesian_trajectory(self, qs, ts):
        xyz = []
        for j in range(len(qs)):
            pino.forwardKinematics(self.pino_model, self.pino_data, qs[j])
            pino.updateFramePlacements(self.pino_model, self.pino_data)
            xyz_pino = copy(self.pino_data.oMf[-1].translation)
            xyz.append(xyz_pino)

        cart_traj = Path()
        cart_traj.header.frame_id = 'F_link_0'
        cart_traj.header.stamp = rospy.Time.now()
        for i in range(len(xyz)):
            point = PoseStamped()
            point.header.stamp = cart_traj.header.stamp + rospy.Duration(ts[i])
            point.pose.position.x = xyz[i][0]
            point.pose.position.y = xyz[i][1]
            point.pose.position.z = xyz[i][2]
            cart_traj.poses.append(point)

        self.cartesian_front_publisher.publish(cart_traj)

    def publish_planner_status(self, planning_time, t):
        planner_status = PlannerStatus()
        planner_status.success = True
        planner_status.planning_time = planning_time * 1000.
        planner_status.planned_motion_time = t[-1]
        self.planner_status_publisher.publish(planner_status)


if __name__ == '__main__':
    node = NeuralPlannerNode()
    rospy.spin()

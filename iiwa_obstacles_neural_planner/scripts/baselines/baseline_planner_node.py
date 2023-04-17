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

import matplotlib.pyplot as plt


BASELINES_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.dirname(BASELINES_DIR)
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)
PLANNING_MODULE_DIR = os.path.join(SCRIPTS_DIR, "manifold_planning")
sys.path.append(PLANNING_MODULE_DIR)
sys.path.append(SCRIPTS_DIR)

from iiwa_obstacles_msgs.msg import PlannerRequest, PlannerStatus
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint, JointTrajectory
from geometry_msgs.msg import Transform, Vector3
from planner_request_utils import unpack_planner_request
from manifold_planning.utils.constants import UrdfModels, Limits, Base
from mpcmpnet_planner import MPCMPNetPlanner
from sst_planner import SSTPlanner
from cbirrt_planner import CBiRRTPlanner
from nlopt_planner import NloptPlanner
from ompl_planner import OMPLPlanner


class BaselinePlannerNode:
    def __init__(self):
        rospy.init_node("baseline_planner_node", anonymous=True)
        controller_type = rospy.get_param("~controllers", "bspline_ff_kino_joint_trajectory_controller")
        self.planning_request_subscriber = rospy.Subscriber("/neural_planner/plan_trajectory", PlannerRequest,
                                                            self.compute_trajectory)
        self.iiwa_front_publisher = rospy.Publisher(f"/{controller_type}/command",
                                                    JointTrajectory,
                                                    queue_size=5)
        self.cartesian_front_publisher = rospy.Publisher(f"/cartesian_trajectory", MultiDOFJointTrajectory,
                                                         queue_size=5)
        self.planner_status_publisher = rospy.Publisher(f"/neural_planner/status", PlannerStatus, queue_size=5)
        self.urdf_path = os.path.join(PLANNING_MODULE_DIR, UrdfModels.iiwa)

        N = 15
        self.pino_model = pino.buildModelFromUrdf(self.urdf_path)
        self.pino_data = self.pino_model.createData()
        print("pino model loaded")
        #self.planner = SSTPlanner(self.pino_model, self.joint_idx)
        #self.planner = MPCMPNetPlanner(self.pino_model, self.joint_idx)
        #self.planner = NloptPlanner(N, self.pino_model, self.joint_idx)
        #self.planner = CBiRRTPlanner(N, self.pino_model, self.joint_idx)
        self.planner = OMPLPlanner("BITstar", N, self.pino_model)
        #self.planner = OMPLPlanner("ABITstar", N, self.pino_model)
        #self.planner = OMPLPlanner("AITstar", N, self.pino_model)
        rospy.sleep(1.)
        print("node loaded")
        self.actual_trajectory = None

    def compute_trajectory(self, msg):
        q_0, q_d, obstacles = unpack_planner_request(msg)

        q, dq, ddq, t, planning_time = self.planner.solve(q_0, q_d, obstacles)

        self.publish_joint_trajectory(q, dq, ddq, t)
        self.publish_cartesian_trajectory(q, t)
        self.publish_planner_status(t, planning_time)


    def publish_joint_trajectory(self, q, dq, ddq, t):
        iiwa_front_msg = JointTrajectory()
        #print(q)
        #print(t)
        #for i in range(6):
        #    plt.subplot(321+i)
        #    plt.plot(q[:, i])
        #plt.savefig("XD.png")
        for i in range(len(q)):
            pt = JointTrajectoryPoint()
            pt.positions = q[i].tolist() + [0.]
            if len(dq):
                pt.velocities = dq[i].tolist() + [0.]
                if len(ddq):
                    pt.accelerations = ddq[i].tolist() + [0.]
            pt.time_from_start = rospy.Duration(t[i])
            iiwa_front_msg.points.append(pt)
        iiwa_front_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        iiwa_front_msg.joint_names = [f"F_joint_{i}" for i in range(1, 8)]
        self.iiwa_front_publisher.publish(iiwa_front_msg)

    def publish_cartesian_trajectory(self, qs, ts):
        assert len(qs) == len(ts)
        cart_traj = MultiDOFJointTrajectory()
        cart_traj.header.frame_id = 'F_link_0'
        z = []
        for i in range(len(qs)):
            pino.forwardKinematics(self.pino_model, self.pino_data, qs[i])
            pino.updateFramePlacements(self.pino_model, self.pino_data)
            xyz_pino = copy(self.pino_data.oMf[-1].translation)
            z.append(xyz_pino[-1])
            point = MultiDOFJointTrajectoryPoint()
            v3 = Vector3(*xyz_pino)
            geometry = Transform(translation=v3)
            point.time_from_start = rospy.Duration(ts[i])
            point.transforms.append(geometry)
            cart_traj.points.append(point)
        #plt.clf()
        #plt.plot(z)
        #plt.savefig("z.png")
        cart_traj.header.stamp = rospy.Time.now()
        self.cartesian_front_publisher.publish(cart_traj)

    def publish_planner_status(self, t, planning_time):
        planner_status = PlannerStatus()
        planner_status.success = True # TODO check if it is really an ok plan, or ignore this field
        planner_status.planning_time = planning_time * 1000.
        planner_status.planned_motion_time = t[-1]
        self.planner_status_publisher.publish(planner_status)


if __name__ == '__main__':
    node = BaselinePlannerNode()
    rospy.spin()

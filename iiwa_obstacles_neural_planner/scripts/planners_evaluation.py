#!/usr/bin/env python
import os.path
import shlex
import signal
import subprocess

import psutil
import rospy
import tf

import numpy as np
from sensor_msgs.msg import JointState

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, DeleteModel, GetModelState, SetModelConfiguration

from iiwa_obstacles_msgs.msg import PlannerRequest, PlannerStatus

from air_hockey_puck_tracker.srv import GetPuckState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from manifold_planning.utils.data import unpack_data_iiwa_obstacles
from gazebo_ros import gazebo_interface
from geometry_msgs.msg import Pose



class PlannersEvaluationNode:
    def __init__(self):
        rospy.init_node("planners_evaluation", anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.planner_request_publisher = rospy.Publisher("/neural_planner/plan_trajectory", PlannerRequest,
                                                         queue_size=5)
        self.robot_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.set_robot_state)
        self.planner_status_subscriber = rospy.Subscriber(f"/neural_planner/status", PlannerStatus,
                                                          self.kill_rosbag_proc)
        self.iiwa_publisher = rospy.Publisher(f"/bspline_ff_kino_joint_trajectory_controller/command",
                                              JointTrajectory,
                                              queue_size=5)
        self.robot_joint_pose = None
        self.robot_joint_velocity = None
        self.rosbag_proc = None
        self.is_moving = False
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_configration_srv = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.delete_model("ground_plane")
        package_path = os.path.join(os.path.dirname(__file__), "..")
        #file_name = os.path.join(package_path, "data/iiwa_obstacles_verysimple3/test/data.tsv")
        #file_name = os.path.join(package_path, "data/iiwa_obstacles_simple1/test/data.tsv")
        #file_name = os.path.join(package_path, "data/iiwa_obstacles_fairlysimple3/test/data.tsv")
        file_name = os.path.join(package_path, "data/iiwa_obstacles_fairlysimple4/test/data.tsv")
        self.data = np.loadtxt(file_name, delimiter="\t").astype(np.float32)  # [:5]
        # self.method = "sst"
        # self.method = "mpcmpnet"
        #self.method = "ours"
        # self.method = "iros"
        # self.method = "nlopt"
        # self.method = "cbirrt"
        #self.method = "ours"
        #self.method = "aitstar_0s"
        self.method = "abitstar_0s"
        #self.method = "bitstar_0s"
        #self.method = "bitstar_10s"

    def move_to(self, q):
        iiwa_front_msg = JointTrajectory()
        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.velocities = [0.] * 7
        pt.accelerations = [0.] * 7
        pt.time_from_start = rospy.Duration(3.0)
        iiwa_front_msg.points.append(pt)
        iiwa_front_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        iiwa_front_msg.joint_names = [f"F_joint_{i}" for i in range(1, 8)]
        self.iiwa_publisher.publish(iiwa_front_msg)
        rospy.sleep(5.)

    def kill_rosbag_proc(self, msg):
        rospy.sleep(max(4., msg.planned_motion_time + 2.))
        if self.rosbag_proc is not None:
            os.kill(self.rosbag_proc.pid, signal.SIGTERM)
        rospy.sleep(1.)
        self.rosbag_proc = None
        self.is_moving = False

    def set_robot_state(self, msg):
        self.robot_joint_pose = msg.position[:7]
        self.robot_joint_velocity = msg.velocity[:7]

    def delete_model(self, name):
        if self.get_model_srv(model_name=name).success:
            self.delete_model_srv(model_name=name)

    def record_rosbag(self, i):
        name = f"data/obs_exp/{self.method}/{i:03d}.bag"
        os.makedirs(os.path.dirname(name), exist_ok=True)
        command = "rosbag record " \
                  f"-O {name} " \
                  "/joint_states /tf " \
                  "/bspline_ff_kino_joint_trajectory_controller/state " \
                  "/neural_planner/status /neural_planner/plan_trajectory"
        # "/bspline_joint_trajectory_controller/state " \
        command = shlex.split(command)
        self.rosbag_proc = subprocess.Popen(command)
        rospy.sleep(1.0)

    def evaluate(self):
        q0, qk, _, _, _, _, obstacles = unpack_data_iiwa_obstacles(self.data)
        #for i in range(0, len(q0)):
        for i in range(0, 10):
            self.delete_balls(int(obstacles[i].shape[0] / 4))
            self.move_to(q0[i])
            self.create_balls(obstacles[i])
            self.record_rosbag(i)
            self.request_plan(qk[i], obstacles[i])
            self.is_moving = True
            k = 0
            while self.is_moving:
                print(self.is_moving, k)
                rospy.sleep(0.1)
                k += 1
                pass
            rospy.sleep(2.)
            #i -= 1

    def delete_balls(self, n):
        for i in range(n):
            self.delete_model(f"ball_{i}")

    def create_balls(self, obstacles):
        def model_xml(x, y, z, r):
            return f"""
            <robot name="ball">
              <link name="world"/>
              <joint name="base_joint" type="fixed">
                <parent link="world"/>
                <child link="ball"/>
                <origin xyz="{x} {y} {z}" rpy="0 0 0"/>
              </joint>
              <link name="ball">
                <inertial>
                  <origin xyz="0 0 0" />
                  <mass value="100.0" />
                  <inertia  ixx="100.0" ixy="0.0"  ixz="0.0"  iyy="100.0"  iyz="0.0"  izz="100.0" />
                </inertial>
                <visual>
                  <origin xyz="0 0 0"/>
                  <geometry>
                    <sphere radius="{r}" />
                  </geometry>
                </visual>
              </link>
              <gazebo reference="ball">
                <material>Gazebo/Blue</material>
              </gazebo>
            </robot>
            """

        obstacles = np.reshape(obstacles, (-1, 4))
        for i, o in enumerate(obstacles):
            gazebo_interface.spawn_urdf_model_client(f"ball_{i}", model_xml(*o), "", Pose(), 'world', "/gazebo")

    def prepare_planner_request(self, q_d, obstacles):
        pr = PlannerRequest()
        pr.q_0 = self.robot_joint_pose
        pr.q_d = q_d.tolist()
        print(obstacles.shape)
        pr.obstacles = obstacles.tolist()
        print(type(pr.obstacles[0]))
        pr.header.stamp = rospy.Time.now()
        return pr

    def request_plan(self, q_d, obstacles):
        pr = self.prepare_planner_request(q_d, obstacles)
        if pr is None:
            return False
        self.planner_request_publisher.publish(pr)
        return True


if __name__ == '__main__':
    node = PlannersEvaluationNode()
    node.evaluate()

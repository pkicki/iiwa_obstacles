from copy import copy
from time import perf_counter

import numpy as np
import pinocchio as pino
from ompl import base as ob
from ompl import geometric as og

from manifold_planning.utils.constants import Limits
from manifold_planning.utils.constants import Robot


class TimeOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si):
        super(TimeOptimizationObjective, self).__init__(si)
        self.si_ = si

    #def motionCost(self, PathLengthOptimizationObjective, *args, **kwargs):
    def motionCost(self, s1, s2):
        expected_time = 0.
        for i in range(self.si_.getStateSpace().getDimension()):
            #expected_time_i = abs(s1[i] - s2[i]) / Limits.q_dot7[i]
            #expected_time = max(expected_time, expected_time_i)
            expected_time = max(expected_time, abs(s1[i] - s2[i]) / Limits.q_dot7[i])
        return expected_time

class OMPLPlanner:
    def __init__(self, planner, n_pts, pino_model):
        self.N = n_pts
        self.M = self.N - 2
        self.D = 7
        self.pino_model = pino_model
        self.pino_data = self.pino_model.createData()

        #self.time = 60.
        self.time = 1.0
        #self.time = 10.0

        rvss = ob.RealVectorStateSpace(self.D)
        bounds = ob.RealVectorBounds(self.D)
        lb = pino_model.lowerPositionLimit[:self.D]
        ub = pino_model.upperPositionLimit[:self.D]
        for i in range(self.D):
            bounds.setLow(i, lb[i])
            bounds.setHigh(i, ub[i])
        rvss.setBounds(bounds)

        self.state_space = rvss
        self.si = ob.SpaceInformation(self.state_space)
        # Create our constraint.
        self.ss = og.SimpleSetup(self.si)

        if planner == "RRTConnect":
            self.planner = og.RRTConnect(self.si)
        elif planner == "BITstar":
            self.planner = og.BITstar(self.si)
        elif planner == "AITstar":
            self.planner = og.AITstar(self.si)
        elif planner == "ABITstar":
            self.planner = og.ABITstar(self.si)
        else:
            print("Unknown planner available are RRTCOnnect, BITstar, AITstar, ABITstar")
            assert False
        self.ss.setPlanner(self.planner)
        self.ss.setup()
        self.si.setup()

    def forward_kinematics(self, q):
        pino.forwardKinematics(self.pino_model, self.pino_data, q)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        xyz_pino = [x.translation for x in self.pino_data.oMf]
        #R_pino = [x.rotation for x in self.pino_data.oMf]
        R_pino = None
        return copy(xyz_pino), copy(R_pino)

    def obstacles(self, x, obstacles):
        allowed_violation = 0.01
        q = np.zeros((self.D))
        for i in range(self.D):
            q[i] = x[i]
        xyz, R = self.forward_kinematics(q)

        xyz = np.stack(xyz, axis=0)
        dists = np.linalg.norm(xyz[np.newaxis] - obstacles[:, np.newaxis, :3], axis=-1)
        radiuses = obstacles[:, -1:] + Robot.radius - allowed_violation
        if np.any(dists < radiuses):
            return False

        return True

    def solve(self, q0, qk, obstacles):
        obstacles = np.reshape(obstacles, (-1, 4))
        self.ss.clear()
        start = ob.State(self.state_space)
        goal = ob.State(self.state_space)
        for i in range(self.D):
            start[i] = q0[i]
        for i in range(self.D):
            goal[i] = qk[i]
        self.ss.setStartAndGoalStates(start, goal, 0.01)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda x: self.obstacles(x, obstacles)))
        self.ss.setOptimizationObjective(TimeOptimizationObjective(self.si))
        self.ss.getOptimizationObjective().setCostThreshold(1000.)
        self.ss.setup()
        stat = self.ss.solve(self.time)
        planning_time = self.ss.getLastPlanComputationTime()
        success = False
        q = np.array([q0])
        t = np.array([0.])
        if stat:
            # Get solution and validate
            path = self.ss.getSolutionPath()
            path.interpolate()
            states = [[x[i] for i in range(self.D)] for x in path.getStates()]
            q = np.array(states)
            success = True
            q_diff = q[1:] - q[:-1]
            diff = np.sum(np.abs(q_diff), axis=-1)
            include = np.concatenate([diff > 0, [True]])
            q = q[include]
            q_diff = q_diff[include[:-1]]
            ts = np.abs(q_diff) / Limits.q_dot7
            t = np.max(ts, axis=-1)
            t = np.concatenate([[0.], t + 1e-4])
            t = np.cumsum(t)
        return q, [], [], t, planning_time

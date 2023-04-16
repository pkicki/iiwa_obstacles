import numpy as np


def unpack_planner_request(msg):
    q_0 = np.array(msg.q_0)
    q_d = np.array(msg.q_d)
    obstacles = np.array(msg.obstacles)
    return q_0, q_d, obstacles

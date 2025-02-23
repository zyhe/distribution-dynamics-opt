"""
## gradient-free vanilla algorithm

Follow the idea of performative prediction, i.e., sampling and optimize

"""
import numpy as np
import sys
sys.path.append("..")
from Models.distribution_dynamics import UserHedge
from .tool_funcs import *


class VanillaGFAlg:
    def __init__(self, sz: float, delta: float):
        """
        :param sz: step size
        :param delta: smoothing radius
        """
        self.sz = sz
        self.delta = delta
        self.obj_prev = 0  # previous objective value, useful for variance reduction
        self.obj_record = 0

    def itr_update(self, dec: np.ndarray, v: np.ndarray, user: UserHedge, budget: float) -> np.ndarray:
        """
        Implement the iterative update
        :param dec: decision vector
        :param v: perturbation vector
        :param user: object of the class UserHedge
        :param budget: total budget on the sum of elements of the decision
        :return: the updated decision and the perturbed decision
        """
        self.obj_record = self.obj_prev
        p_cur = user.p_cur
        dim = dec.shape[0]
        # recover the perturbed decision
        dec_pert = dec + self.delta * v
        # evaluate the objective
        obj = p_cur.T @ dec_pert + user.w_entropy * (p_cur.T @ np.log(p_cur))
        grad_est = dim * (obj - self.obj_prev) * v / self.delta
        # grad_est = dim * obj * v / self.delta

        self.obj_prev = obj  # record the objective value

        # iterative update
        dec_cur = proj_simplex(dec + self.sz * grad_est, budget)
        return dec_cur

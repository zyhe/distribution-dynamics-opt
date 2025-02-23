"""
## vanilla algorithm

Follow the idea of performative prediction, i.e., sampling and optimize

"""
import numpy as np
import sys
sys.path.append("..")
from Models.distribution_dynamics import UserHedge
from .tool_funcs import *


class VanillaAlg:
    def __init__(self, sz: float):
        """
        :param sz: step size
        """
        self.sz = sz

    def itr_update(self, dec: np.ndarray, user: UserHedge, budget: float) -> np.ndarray:
        """
        Implement the iterative update
        :param dec: decision vector
        :param user: object of the class UserHedge
        :param budget: total budget on the sum of elements of the decision
        # :param penalty_coeff: penalty parameter
        """
        # constr_vio = max(0, lbd - np.sum(dec_prev))
        # constr_vio = lbd - np.sum(dec_prev)
        # grad = user.p_cur - penalty * constr_vio * np.ones_like(dec_prev)
        # dec_cur = proj_box(dec_prev - self.sz * grad, np.zeros_like(dec_prev), bd_dec * np.ones_like(dec_prev))
        # grad = - user.p_cur + penalty_coeff * dec_prev

        grad = user.p_cur
        dec_cur = proj_simplex(dec + self.sz * grad, budget)
        return dec_cur

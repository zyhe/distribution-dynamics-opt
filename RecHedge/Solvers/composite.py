"""
## composite algorithm

Use the dynamics information to construct the gradient

"""
import numpy as np
import sys
sys.path.append("..")
from Models.distribution_dynamics import UserHedge
from .tool_funcs import *


class CompositeAlg:
    def __init__(self, sz: float):
        """
        :param sz: step size
        """
        self.sz = sz
    
    def sens_mat(self, dec: np.ndarray, user: UserHedge) -> np.ndarray:
        """
        Construct the sensitivity matrix
        :param dec: decision vector
        :param loss: loss vector
        :return: the Jacobian matrix
        """
        # Stabilize exponentials by subtracting the max value of the scaled decision
        max_loss = np.max(-user.epsilon * dec)
        z = np.exp(-user.epsilon * dec - max_loss)
        sum_z = np.sum(z, axis=0)

        diag_z = np.diag(z.ravel())
        jacobian = user.lambda2 / (1-user.lambda1) * -user.epsilon * (sum_z * diag_z - z @ z.T) / (sum_z**2)
        return jacobian

    def itr_update(self, dec: np.ndarray, user: UserHedge, budget: float) -> np.ndarray:
        """
        Implement the iterative update
        :param dec: decision vector
        :param user: object of the class UserHedge
        :param budget: total budget on the sum of elements of the decision
        # :param penalty_coeff: penalty parameter
        """
        p_cur = user.p_cur
        partial_grad_p = dec + user.w_entropy * (np.ones_like(p_cur) + np.log(p_cur))
        grad = user.p_cur + self.sens_mat(dec, user) @ partial_grad_p
        # grad = -user.p_cur - self.sens_mat(dec, user) @ partial_grad_p + penalty_coeff * dec
        dec_cur = proj_simplex(dec + self.sz * grad, budget)
        return dec_cur

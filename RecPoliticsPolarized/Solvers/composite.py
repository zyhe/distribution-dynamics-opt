"""
## composite algorithm

Use the dynamics information to construct the gradient

"""
import numpy as np
from .tool_funcs import *


class CompositeAlg:
    def __init__(self, sz, num_sample, num_itr, radi, dyn_param):
        """
        :param sz: step size
        :param num_sample: number of samples used in stochastic gradients
        :param num_itr: number of iterations
        :param radi: size of the norm ball
        :param dyn_param: parameters in dynamics (i.e., lambda_p, sigma)
        """
        self.sz = sz
        self.num_sample = num_sample
        self.num_itr = num_itr
        self.radi = radi
        self.sigma = dyn_param["sigma"]  # coefficient before the inner product term
        self.lambda_p = dyn_param["lambda_p"]  # coefficient of combination
        self.id_mat = np.eye(dyn_param["dim"])
        self.radi_inv = 1 / radi

    def _grad_construct(self, state_pop, state_norm, dec) -> np.ndarray:
        """
        construct the gradient
        :param state_pop: state of population (dim * size_pop)
        :param state_norm: norm of the state of population
        :param dec: decision vector
        """
        size_pop = state_pop.shape[1]
        # extract samples from the population
        idx = np.random.choice(size_pop, size=self.num_sample)

        # # use the full sample
        # idx = np.arange(0, size_pop)

        # construct the gradient
        # objective: p^T*q -> composite gradient: p + sens_mat*q
        state_pop_sample = state_pop[:, idx]
        state_norm_sample = state_norm[:, idx]

        # first term of the stochastic gradient
        grad_first_term = np.mean(state_pop_sample, axis=1, keepdims=True)  # size: dim * 1, corresponding to p

        # second term of the stochastic gradient
        sens_mat_pop_mean = self._sens_mat_construct(state_pop_sample, state_norm_sample, dec)  # size: dim * dim
        grad_second_term = sens_mat_pop_mean @ dec  # size: dim * 1

        # obtain the gradient
        grad = grad_first_term + grad_second_term  # size: dim * 1
        return grad

    def _sens_mat_construct(self, state_pop, state_norm, dec) -> np.ndarray:
        """
        construct the sensitivity matrix of each individual in the population
        :param state_pop: state of population (dim * size_pop)
        :param state_norm: norm of the state of population (1 * size_pop)
        :param dec: decision vector (dim * 1)
        :return: sens_mat_pop: sensitivities of the population (size_pop * dim * dim)
        """
        dim, size_pop = state_pop.shape

        # Initialize the sensitivity matrix accumulator
        sens_mat_sum = np.zeros((dim, dim))

        # precompute to save efforts
        dec_out_prod = dec @ dec.T

        for i in range(size_pop):
            p_cur = state_pop[:, i:i + 1]
            p_T_mid = self.radi * self.id_mat - self.radi_inv * p_cur @ p_cur.T  # r*I - (1/r)*p*p^T
            inv_mat_mid = np.linalg.inv(self.lambda_p * p_T_mid + self.sigma * dec_out_prod @ p_T_mid
                                        - state_norm[0, i] * self.id_mat)
            sens_mat_sum += - (self.sigma * ((p_cur.T @ dec) * self.id_mat + p_cur @ dec.T)
                               @ p_T_mid @ inv_mat_mid)

        sens_mat_pop_mean = sens_mat_sum / size_pop
        return sens_mat_pop_mean

    def itr_update(self, pop, dec_prev, state_norm) -> np.ndarray:
        """
        iterative update of the algorithm
        :param pop: an object of population
        :param dec_prev: previous decision
        :param state_norm: norm of the state of population
        :return decision
        """
        # gradient ascent for utility optimization
        grad = self._grad_construct(pop.state_cur, state_norm, dec_prev)
        dec_cur = dec_prev + self.sz * grad
        # project to the l2 norm ball
        dec_cur = proj_ball(dec_cur, self.radi)
        return dec_cur

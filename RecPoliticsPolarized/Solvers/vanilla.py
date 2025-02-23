"""
## vanilla algorithm

Follow the idea of performative prediction, i.e., sampling and optimize

"""
import numpy as np
from .tool_funcs import *



class VanillaAlg:
    def __init__(self, sz, num_sample, num_itr, radi):
        """
        :param sz: step size
        :param num_sample: number of samples used in stochastic gradients
        :param num_itr: number of iterations
        :param radi: size of the norm ball
        """
        self.sz = sz
        self.num_sample = num_sample
        self.num_itr = num_itr
        self.radi = radi

    def _grad_construct(self, state_pop) -> np.ndarray:
        """
        construct the gradient
        :param state_pop: state of population (dim * size_pop)
        """
        # extract samples from the population
        # use stochastic samples to construct the gradient
        size_pop = state_pop.shape[1]
        # extract samples from the population
        idx = np.random.randint(0, size_pop, size=self.num_sample)

        # # use the full sample
        # idx = np.arange(0, size_pop)

        # objective: p^T*q -> vanilla gradient: p
        sample_selected = state_pop[:, idx]
        # calculate the mean of each row and obtain the gradient
        grad = np.mean(sample_selected, axis=1, keepdims=True)  # size: dim * 1
        return grad

    def itr_update(self, pop, dec_prev, state_norm) -> np.ndarray:
        """
        iterative update of the algorithm
        :param pop: an object of population
        :param dec_prev: previous decision
        :param state_norm: norm of the state of population (not useful in this algorithm)
        :return decision
        """
        # gradient ascent for utility optimization
        grad = self._grad_construct(pop.state_cur)
        dec_cur = dec_prev + self.sz * grad
        # project to the l2 norm ball
        dec_cur = proj_ball(dec_cur, self.radi)
        return dec_cur

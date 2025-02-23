"""
Define the class to analyze the closed-loop response of the system
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

import sys
sys.path.append("..")
from Models.population import Population
from Solvers.composite import CompositeAlg
from Solvers.vanilla import VanillaAlg
from Solvers.tool_funcs import *

# Configure font settings
plt.rcParams.update({
    "font.family": ["Times New Roman", "serif"],  # another option "serif"
    "font.size": 14,
    "mathtext.fontset": "cm",  # Computer Modern for math
})


class CLResponse:
    """
    Class of the closed-loop response
    """

    def __init__(self,
                 dim: int,
                 radi: int,
                 size_pop: int,
                 angle_bd: float,
                 sigma,
                 lambda_p,
                 sz,
                 num_sample: int,
                 num_itr: int,
                 num_trial: int,
                 mode_dict: dict
                 ):

        # parameters of the problem and the population
        self.dim = dim  # dimension of the decision variable & preference vector
        self.radi = radi  # radius of the norm ball
        self.size_pop = size_pop  # number of agents in the population
        self.angle_bd = angle_bd  # lower bound on the relative angle

        # parameters of the polarized model
        # lambda_p * state_prev + (1 - lambda_p) * state_init + sigma * inner_prod * dec
        # followed by normalization
        self.sigma = sigma
        self.lambda_p = lambda_p

        # parameters of the algorithms
        self.sz = sz  # step size
        self.num_sample = num_sample  # number of samples used per iteration
        self.num_itr = num_itr  # number of iterations
        self.num_trial = num_trial  # number of independent trials

        # dictionary containing the modes
        self.mode_dict = mode_dict

        # initialize the solver
        self.alg = None
        # initial the decision vector, which is normalized according to the radius
        dec_raw = np.random.randn(self.dim, 1)
        self.dec_init = dec_raw * (self.radi / np.linalg.norm(dec_raw))

        # initialize the population
        self.pop = Population(self.dim, self.radi, self.size_pop, self.angle_bd,
                              self.sigma, self.lambda_p, self.dec_init)
        self.pop.population_init(self.mode_dict["data_dist"])  # "uniform", "bimodal", or "load"

        # store results
        # self.dec_data = np.zeros((self.dim, self.num_itr))  # data matrix of decisions
        self.utility_dynamic_data = np.zeros((self.num_trial, self.num_itr))  # real-time population-wide utilities
        self.utility_ss_data = np.zeros((self.num_trial, self.num_itr))  # steady-state population-wide utilities
        self.dist_pt_data = np.zeros((self.num_trial, self.num_itr))  # distances to the optimal point
        self.dec_final_data = np.zeros((self.dim, self.num_trial))  # final decisions
        self.utility_final_data = np.zeros(self.num_trial)  # final utilities
        self.angle_final_data = np.zeros((self.num_trial, self.size_pop))  # final angles

        # index of the elements while analyzing population-wide positions
        self.hist_idx = np.sort(np.random.choice(self.dim, size=4, replace=False))

    def _select_solver(self):
        """Select the solver based on the given mode."""
        if self.mode_dict["solver"] == "vanilla":
            return VanillaAlg(self.sz, self.num_sample, self.num_itr, self.radi)
        elif self.mode_dict["solver"] == "composite":
            # collect the dynamics parameters
            dyn_param = {"sigma": self.sigma, "lambda_p": self.lambda_p, "dim": self.dim}
            return CompositeAlg(self.sz, self.num_sample, self.num_itr, self.radi, dyn_param)
        else:
            raise ValueError(f"Unsupported solver type: {self.mode_dict['solver']}")

    def response(self):
        """
        Response of the closed-loop system
        """
        logging.info(f'Start running the {self.mode_dict["solver"]} algorithm.')
        self.alg = self._select_solver()
        opt_pt = self.pop.opt_dec  # store the optimal point outside the loop

        # closed-loop response
        for trial in range(self.num_trial):
            dec = self.dec_init  # start from the same initial decision
            self.pop.angle_init = angle_pop_calculate(dec, self.pop.state_init)
            for i in range(self.num_itr):
                # dynamics
                state_norm = self.pop.per_step_dynamics(dec)

                # store the current decision and the corresponding performance metrics
                self.utility_dynamic_data[trial, i] = self.pop.utility_cur(dec)
                self.utility_ss_data[trial, i] = self.pop.utility_ss(dec)
                self.dist_pt_data[trial, i] = np.linalg.norm(dec - opt_pt)
                # self.utility_ss[i] = self.pop.utility_ss(dec)
                # self.dec_data[:, i:i + 1] = dec

                # obtain the new decision
                dec = self.alg.itr_update(self.pop, dec, state_norm)

            # store and print solutions
            self.dec_final_data[:, trial:trial+1] = dec  # size: dim * 1
            self.utility_final_data[trial] = self.utility_ss_data[trial, -1]
            # self.utility_final_data[trial] = self.utility_dynamic_data[trial, -1]
            self.angle_final_data[trial] = angle_pop_calculate(dec, self.pop.state_cur)
            self.pop.state_final = self.pop.state_cur

            # reset the population after each trial
            self.pop.population_reset()

        # reorganize data of final angles
        self.pop.angle_final = self.angle_final_data.reshape(-1)

        # visualize
        # self.visualize_conv()
        self.visualize_hists(self.mode_dict["solver"])

        logging.info(f'Finish running the {self.mode_dict["solver"]} algorithm.')
        logging.info(f'The decision vector is {self.dec_final_data[:, -1].T} '
                     f'with norm {np.linalg.norm(self.dec_final_data[:, -1])}.')
        logging.info(f'The final utility averaged over trials is {np.mean(self.utility_final_data)}.')
        # logging.info(f'The nominal steady-state utility is {self.pop.utility_ss(self.dec_final)}.')

    def response_opt_dec(self):
        """
        Response of the closed-loop system with the fixed optimal decision vector
        """
        logging.info(f'Demonstrate the performance of the optimal decision.')
        self.pop.angle_init = angle_pop_calculate(self.dec_init, self.pop.state_init)

        # rollout, i.e., use the same decision
        itr_cnt = 15
        for _ in range(itr_cnt):
            _ = self.pop.per_step_dynamics(self.pop.opt_dec)

        self.pop.state_final = self.pop.state_cur
        self.pop.angle_final = angle_pop_calculate(self.pop.opt_dec, self.pop.state_cur)

        self.visualize_hists("optimal")
        # reset the population
        self.pop.population_reset()

    # def visualize_conv(self, data):
    #     """
    #     Plot the evolution of the utility/convergence measure
    #     :param data: data of the utility/convergence measure
    #     """
    #     plt.figure()
    #     plt.plot(np.arange(self.num_itr), data, linewidth=2, label=self.mode_dict["solver"])
    #     plt.xlabel('Number of Iterations')
    #     plt.ylabel('Utilities')
    #     plt.grid()
    #     # plt.show(block=False)

    def visualize_hists(self, mode: str):
        """
        Plot the histograms of positions and angles between positions and decisions
        :param mode: "stochastic", "composite", or "optimal"
        """
        # show the histogram of positions
        # for pos_id in range(self.dim):
        #     self.pop.histogram(mode, pos_id)

        for pos_id in self.hist_idx:
            self.pop.histogram(mode, pos_id)

        # show the histogram of population-wide angles
        self.pop.angle_histogram(mode)


def construct_mode_dict(solver, distribution="load"):
    return {
        "solver": solver,  # "composite" or "vanilla"
        "data_dist": distribution,  # "uniform", "bimodal", or "load"
    }


def angle_pop_calculate(dec: np.ndarray, state_mat: np.ndarray) -> np.ndarray:
    """
    Calculate the angles between preferences and the decision
    :param dec: the decision vector (size: dim * 1)
    :param state_mat: matrix containing the states (i.e., preferences) of the population
    :return: angle_data: array containing the angles (size: (size_pop,))
    """
    state_norm = np.linalg.norm(state_mat, axis=0, keepdims=True)
    dec_norm = np.linalg.norm(dec)
    inner_prod = dec.T @ state_mat
    cos_angles = ((inner_prod / state_norm) / dec_norm).ravel()  # obtain a 1D array
    return np.rad2deg(np.arccos(np.clip(cos_angles, -1.0, 1.0)))  # unit: degree

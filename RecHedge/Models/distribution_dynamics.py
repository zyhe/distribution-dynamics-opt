"""
## Distribution dynamics

A user following distribution dynamics with a modified hedge structure

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cyipopt import minimize_ipopt
# from autograd import grad
# import jax
# import jax.numpy as jnp
from typing import Tuple
import logging


class UserHedge:
    def __init__(self, dim: int, lambda1: float, lambda2: float, epsilon: float, budget: float,
                 w_entropy: float, dec_init: np.ndarray):
        """
        :param dim: dimension of the state, i.e., the number of choices
        :param lambda1: combination coefficient in the modified hedge dynamics
        :param lambda2: combination coefficient in the modified hedge dynamics
        :param epsilon: step size/coefficient in the hedge dynamics
        :param budget: total budget of the loss vector
        :param w_entropy: weight corresponding to the entropy regularization
        :param dec_init: initial guess for the optimal decision
        """
        # problem and dynamics parameters
        self.dim = dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
        self.budget = budget
        self.w_entropy = w_entropy

        # preference state
        # self.p_init = normalize_simplex(np.ones((dim, 1)))  # uniform initial distribution
        self.p_init = normalize_simplex(np.random.rand(dim, 1))
        self.p_cur = self.p_init

        # calculate the optimal solution via feedforward numerical optimization
        self.opt_pt, self.opt_val = self.maximize_obj(dec_init.flatten(), solver='ipopt')  # 'scipy' or 'ipopt'
        self.opt_ss = self.eval_opt_dec()  # steady-state distribution corresponding to the optimal decision
        pass

    def per_step_dynamics(self, dec: np.ndarray) -> np.ndarray:
        """
        Implement the modified hedge dynamics at every time step
        :param: dec: the current loss vector, which serves as the input
        :return: the preference state at the next time step
        """
        # convex combination
        self.p_cur = (self.lambda1 * self.p_cur + self.lambda2 * self.softmax_vec(dec)
                      + (1 - self.lambda1 - self.lambda2) * self.p_init)
        # uncomment the following line if we drop the initial distribution
        # self.p_cur = self.lambda1 * self.p_cur + self.lambda2 * self.softmax_vec(dec)
        return self.p_cur

    def softmax_vec(self, dec: np.ndarray) -> np.ndarray:
        """
        Calculate the softmax function of the given decision (i.e., loss vector)
        Formula: exp(-epsilon * dec) / sum(exp(-epsilon * dec))
        :param dec: the current loss vector, which serves as the input
        :return: vector from the softmax function
        """
        dec_shift = -dec - np.max(-dec)  # shift for numerical stability
        exp_weight = np.exp(self.epsilon * dec_shift)
        exp_weight /= np.sum(exp_weight)
        return exp_weight

    def steady_state(self, dec: np.ndarray) -> np.ndarray:
        """
        Calculate the steady-state preference state corresponding to the given decision
        :param dec: decision vector
        :return: steady-state preference state
        """
        p_ss = ((self.lambda2 * self.softmax_vec(dec) + (1 - self.lambda1 - self.lambda2) * self.p_init)
                / (1 - self.lambda1))
        # uncomment the following line if we drop the initial distribution
        # p_ss = self.lambda2 * self.softmax_vec(dec) / (1 - self.lambda1)
        return p_ss

    # def naive_dec_utility(self) -> np.ndarray:
    #     """
    #     Evaluate the steady-state utility corresponding to the naive decision
    #     :return: steady-state utility of the naive decision
    #     """
    #     # Use a greedy decision, i.e., allocating all the budget to the most probable element
    #     dec_naive = self.budget * (self.p_init == np.max(self.p_init)).astype(int)
    #     return self.steady_state(dec_naive).T @ dec_naive

    # def gradient_obj(self, dec: np.ndarray) -> np.ndarray:
    #     grad_func = grad(self.objective_function)
    #     return grad_func(dec)

    def objective_function(self, dec: np.ndarray) -> float:
        """The objective function used by optimization solvers"""
        dec = dec.reshape(-1, 1)
        steady_state = self.steady_state(dec)
        # Negative because we are maximizing
        # the objective is the sum of an inner product (expected loss) and an entropic regularizer
        obj_val = (-steady_state.T @ dec).item() - self.w_entropy*np.sum(steady_state * np.log(steady_state))
        return obj_val

    def maximize_obj(self, x0: np.ndarray, solver: str) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function (i.e., the inner product) through scipy or IPOPT
        :param x0: initial guess
        :param solver: the solver to use, either 'scipy' or 'ipopt'
        :return: optimal decision and optimal value
        """

        # Define the objective function used by optimization solvers

        def objective_function(dec: np.ndarray) -> float:
            """The objective function used by optimization solvers"""
            dec = dec.reshape(-1, 1)
            steady_state = self.steady_state(dec)
            # Negative because we are maximizing
            # the objective is the sum of an inner product (expected loss) and an entropic regularizer
            obj_val = (-steady_state.T @ dec).item() - self.w_entropy * np.sum(steady_state * np.log(steady_state))
            return obj_val

        # Define the constraint function
        def constraint_sum(dec: np.ndarray) -> float:
            """The constraint function used by optimization solvers"""
            return np.sum(dec) - self.budget

        # Constraints
        constraints = {'type': 'eq', 'fun': constraint_sum}

        # Bounds for decision variables
        bounds = [(0, None) for _ in range(self.dim)]

        # Optimize
        if solver == 'scipy':
            result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Use IPOPT
            result = minimize_ipopt(objective_function, x0, bounds=bounds, constraints=constraints,
                                    options={'acceptable_tol': 1e-6, 'tol': 1e-7})
            # Manually set success to True if acceptable tolerance was reached
            if result.status == 1:  # Status 1 indicates acceptable tolerance reached
                result.success = True

        if result.success:
            optimal_dec = result.x
            optimal_value = -result.fun  # Negate to get the original objective value
            logging.info(f'The optimal decision is {optimal_dec} with the optimal value {optimal_value}.')
            return optimal_dec.reshape(-1, 1), optimal_value
        else:
            logging.error("Optimization failed.")
            return None, None

    def eval_opt_dec(self) -> np.ndarray:
        """
        Evaluate the steady-state distribution and the structure of the objective given the optimal decision
        :return: the steady-state distribution corresponding to the optimal decision
        """
        opt_ss = self.steady_state(self.opt_pt)
        logging.info(f'The steady-state distribution given the optimal decision is {opt_ss.ravel()}')
        logging.info(f'In the objective, the inner product (expected value) is {(opt_ss.T @ self.opt_pt).item()} '
                     f'and the regularizer is {self.w_entropy * np.sum(opt_ss * np.log(opt_ss))}')
        return opt_ss

    def reset(self):
        """Reset the preference state"""
        self.p_cur = self.p_init


def normalize_simplex(p_mat: np.ndarray) -> np.ndarray:
    """
    Normalize preference states so that they lie in the probability simplex
    :param p_mat: state matrix
    :return: normalized state matrix
    """
    sum_column = np.sum(p_mat, axis=0, keepdims=True)
    p_mat /= sum_column
    return p_mat

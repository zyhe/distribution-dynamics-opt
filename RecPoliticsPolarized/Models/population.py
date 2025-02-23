"""
## User population

Population equipped with preference dynamics

"""
import logging
import numpy as np
import matplotlib.pyplot as plt
# import jax
# import jax.numpy as jnp
# # import gurobipy as gp
# from gurobipy import GRB
from pathlib import Path
# from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from cyipopt import minimize_ipopt
from typing import Tuple


# Configure font settings
plt.rcParams.update({
    "font.family": ["Times New Roman", "serif"],  # another option "serif"
    "font.size": 14,
    "mathtext.fontset": "cm",  # Computer Modern for math
})

# Configure logging
logging.basicConfig(
    filename='cl_simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Define a dictionary to map modes to color codes
MODE_COLORS = {
    "vanilla": "#1f77b4",  # Blue
    "composite": "#d62728",  # red
    "optimal": "#ff7f0e",  # orange  #"#9467bd",  # purple
    "basic": "#76B7B2"  # muted green
}

ORDINAL_DICT = {
    0: "0th", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
    6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
    11: "11th", 12: "12th", 13: "13th", 14: "14th",
    15: "15th", 16: "16th", 17: "17th", 18: "18th",
    19: "19th", 20: "20th"
}


class Population:
    def __init__(self, dim: int, radi: float, size_pop: int, angle_bd: float,
                 sigma: float, lambda_p: float, dec_init: np.ndarray):
        """
        :param dim: dimension of the preference vector
        :param radi: radius of the norm ball
        :param size_pop: size of the population
        :param angle_bd: lower bound on the relative angle, admissible range: [angle_bd, 180]
        :param sigma: constant step size in the polarized dynamics
        :param lambda_p: combination coefficient in the polarized dynamics
        :param dec_init: initial guess of the optimal decision
        """
        self.dim = dim
        self.radi = radi
        self.size_pop = size_pop
        self.angle_bd = angle_bd
        self.state_init = None  # initial preference states, dim * size_pop
        self.state_cur = None  # current preference states, dim * size_pop
        self.state_final = None  # final preference states, dim * size_pop
        self.init_avg = None  # average initial state, dim * 1
        self.angle_init = None  # angles between the decision and initial states, (size_pop,)
        self.angle_final = None  # angles between the decision and current states, (size_pop,)
        self.dec_init = dec_init  # initial guess, dim * 1

        # variables corresponding to the polarized model
        self.sigma = sigma  # coefficient before the inner product term
        self.lambda_p = lambda_p  # coefficient of combination

        # Store the gradient function
        # self.grad_utility_ss = jax.grad(self.utility_ss)
        
        # calculate the optimal decision and the optimal value
        self.opt_dec = None
        self.opt_val = None

        # specify the directories for storing histograms
        # the path is relative to closed_loop_simulation.py, which calls this function
        self.hist_dir = Path(f'Figures/Histograms/bd-{self.angle_bd}-lambda-{self.lambda_p}-sigma-{self.sigma}')
        self.hist_dir.mkdir(parents=True, exist_ok=True)

    def population_init(self, mode):
        """
        set the initial preference states of the population
        :param mode: "uniform", "bimodal", or "load"
        """
        if mode == "uniform":
            self.state_init = self._initialize_uniform_restricted(self.angle_bd)
            # self.state_init = self._initialize_uniform()
        elif mode == "bimodal":
            self.state_init = self._initialize_bimodal()
        elif mode == "load":
            self.state_init = self._initialize_from_file()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # set the current state
        self.state_cur = self.state_init
        # calculate the average of the initial states (the mean of every row)
        self.init_avg = np.mean(self.state_init, axis=1, keepdims=True)  # dim * 1
        # calculate the optimal solution
        self.opt_dec, self.opt_val = self.maximize_obj(self.dec_init, solver="ipopt")  # solver: "scipy" or "ipopt"

    def population_reset(self):
        """
        reset the state of the overall population
        """
        self.state_cur = self.state_init

    def per_step_dynamics(self, dec: np.ndarray) -> np.ndarray:
        """
        simulate the polarized preference dynamics
        :param dec: decision/policy (size: dim * 1)
        :return state_norm: norm of the state of each individual
        """
        state_unscaled = self.lambda_p * self.state_cur + (1 - self.lambda_p) * self.state_init \
                         + self.sigma * dec @ (dec.T @ self.state_cur)
        # normalization
        state_norm = np.linalg.norm(state_unscaled, axis=0, keepdims=True)  # size: 1* size_pop
        self.state_cur = self.radi * state_unscaled / state_norm
        return state_norm

    def histogram(self, mode: str, pos_id: int = 1):
        """
        Plot the histogram of preferences
        :param mode: "vanilla", "composite", or "optimal"
        :param pos_id: the index of the element of the preference vector
        """
        plt.figure()
        plt.hist(self.state_init[pos_id, :], alpha=0.4, label='Initial',
                 weights=np.ones_like(self.state_init[pos_id, :]) / len(self.state_init[pos_id, :]),
                 color=MODE_COLORS["basic"])  # initial
        # reason of using np.ones_like: give equal weights to samples
        # bins = 30
        plt.hist(self.state_final[pos_id, :], alpha=0.6, label='Final',
                 weights=np.ones_like(self.state_final[pos_id, :]) / len(self.state_final[pos_id, :]),
                 color=MODE_COLORS[mode])  # current
        # # Plot the KDE as an estimate of the PDF
        # kde = gaussian_kde(self.state_cur[0, :])
        # x = np.linspace(min(self.state_cur[0, :]), max(self.state_cur[0, :]), 1000)
        # plt.plot(x, kde.pdf(x), color='red', linewidth=2, label='PDF')
        plt.legend()
        plt.xlabel(f'Position ({ORDINAL_DICT[pos_id + 1]} element)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of preferences ({mode})')
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.hist_dir / f'pos_histogram_{mode}_{pos_id}.png',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    def angle_histogram(self, mode: str):
        """
        Plot the histogram of angles between preferences and the decision
        :param mode: "stochastic", "composite", or "optimal"
        """
        plt.figure()
        plt.hist(self.angle_init, alpha=0.4, label='Initial',
                 # weights=np.ones_like(self.angle_init) / len(self.angle_init),
                 color=MODE_COLORS["basic"], bins=80, density=True)  # initial
        # bins = 30
        plt.hist(self.angle_final, alpha=0.6, label='Final',
                 # weights=np.ones_like(self.angle_final) / len(self.angle_final),
                 color=MODE_COLORS[mode], bins=80, density=True)  # current
        # # Plot the KDE as an estimate of the PDF
        # kde = gaussian_kde(angle_data_cur)
        # x = np.linspace(min(angle_data_cur), max(angle_data_cur), 1000)
        # plt.plot(x, kde.pdf(x), color='red', linewidth=2, label='PDF')
        plt.legend()
        plt.xlabel('Angle ($^{\circ}$)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of preference-decision angles ({mode})')
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.hist_dir / f'agl_histogram_{mode}.png',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    # functions related to convergence measures
    # real-time objective value
    def utility_cur(self, dec: np.ndarray) -> float:
        """
        calculate the current population-wide utility (non-steady-state)
        :param dec: decision (dim * 1)
        :return: scalar utility (the average of the current population-wide utility)
        """
        utility_pop = dec.T @ self.state_cur  # size: 1 * size_pop
        return np.mean(utility_pop)

    # steady-state objective value
    def utility_ss(self, dec: np.ndarray) -> float:
        """
        Evaluate the steady-state utility given a fixed decision
        :param dec: decision (dim * 1)
        :return: scalar utility (the average of the current population-wide utility)
        """
        state_cur_copy = self.state_cur.copy()
        # reset the current state of the population
        self.state_cur = self.state_init

        # rollout, i.e., use the same decision
        itr_cnt = 10
        for _ in range(itr_cnt):
            _ = self.per_step_dynamics(dec)
        utility_ss_val = self.utility_cur(dec)

        # set back the current state
        self.state_cur = state_cur_copy
        return utility_ss_val
    
    def maximize_obj(self, x0: np.ndarray, solver: str) -> Tuple[np.ndarray, float]:
        """
        Optimize the population-wide objective function through scipy or IPOPT
        :param x0: initial guess
        :param solver: the solver to use, either 'scipy' or 'ipopt'
        :return: optimal decision and optimal value
        """
        # Define the objective function used by optimization solvers
        def objective_function(dec: np.ndarray) -> float:
            # convert the input (i.e., an 1D array) of solvers
            dec = dec.reshape(-1, 1)
            return -self.utility_ss(dec)  # negative because of maximization
        
        # Obtain the gradient function
        # def gradient_function(dec: np.ndarray) -> np.ndarray:
        #     # Solvers expect a 1D array
        #     dec = jnp.asarray(dec.reshape(-1, 1))
        #     grad_jnp = -self.grad_utility_ss(dec)  # negative because of maximization
        #     # Convert jnp.ndarray to np.ndarray
        #     return np.asarray(grad_jnp)
        
        # Define the constraint function
        def constraint_function(dec: np.ndarray) -> float:
            return self.radi - np.linalg.norm(dec)
        
        # Define the constraints dictionary
        constraints = {'type': 'ineq', 'fun': constraint_function}
        
        # Bounds for decision variables
        bounds = [(-self.radi, self.radi) for _ in range(self.dim)]

        # convert the initial guess to an 1D array
        x0_flat = x0.flatten()

        # Optimize
        if solver == 'scipy':
            # use scipy
            result = minimize(objective_function, x0_flat, method='SLSQP', bounds=bounds,
                              constraints=constraints)  # jac=gradient_function,
        elif solver == 'ipopt':
            # use IPOPT
            result = minimize_ipopt(objective_function, x0_flat, bounds=bounds, constraints=constraints,
                                    options={'acceptable_tol': 1e-7, 'tol': 1e-8, 'constr_viol_tol': 1e-6})
            # jac=gradient_function,
            # Manually set success to True if acceptable tolerance was reached
            if result.status == 1:  # Status 1 indicates acceptable tolerance reached
                result.success = True
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if result.success:
            optimal_dec = result.x
            optimal_value = -result.fun  # Negate to get the original objective value
            logging.info(f'{solver}: The optimal decision is {optimal_dec} with the optimal value {optimal_value}.')
            return optimal_dec.reshape(-1, 1), optimal_value
        else:
            logging.error(f"Optimization failed: {result.message}")
            raise RuntimeError("Optimization failed")

    # def ground_truth(self):
    #     """
    #     Use Gurobi to calculate the optimal solution
    #     It is possible when we consider linear FJ dynamics
    #     """
    #     opt_model = gp.Model("rec_politics_model")
    #     opt_model.Params.NonConvex = 2
    #
    #     # define the decision variable
    #     q = opt_model.addVars(self.dim, vtype=GRB.CONTINUOUS, name="q")
    #
    #     # quadratic part of the objective
    #     quad_expr = gp.QuadExpr()
    #     for i in range(self.dim):
    #         for j in range(self.dim):
    #             # if self.ss_dec[i, j] != 0:
    #             quad_expr.add(self.ss_dec[j, i] * q[i] * q[j])
    #
    #     # linear part of the objective
    #     # we need the average of the preference (i.e., the mean of the rows of state_init)
    #     lin_coeff = self.ss_init_state @ np.mean(self.state_init, axis=1)
    #     linear_expr = gp.LinExpr()
    #     for i in range(self.dim):
    #         # if lin_coeff[i] != 0:
    #         linear_expr.add(lin_coeff[i] * q[i])
    #
    #     # set the objective
    #     opt_model.setObjective(quad_expr + linear_expr, GRB.MAXIMIZE)
    #
    #     # add the norm constraint
    #     norm_expr = gp.QuadExpr()
    #     for i in range(self.dim):
    #         norm_expr.add(q[i] * q[i])
    #     opt_model.addQConstr(norm_expr <= self.radi**2, "norm ball")
    #
    #     # optimize the model
    #     opt_model.optimize()
    #
    #     # obtain the solutions
    #     if opt_model.status == GRB.OPTIMAL:
    #         opt_pt = np.zeros(self.dim)
    #         for i, v in enumerate(opt_model.getVars()):
    #             print(f"{v.VarName} {v.X:g}")
    #             opt_pt[i] = v.X
    #         # print the optimal objective value
    #         print(f"Optimal objective value: {opt_model.objVal}")
    #         # store the optimal point & value
    #         self.opt_pt = opt_pt
    #         self.opt_val = opt_model.objVal
    #         return None
    #     else:
    #         print("No optimal solution found.")
    
    # def sec_moment(self, dec: jnp.ndarray) -> float:
    #     """
    #     Calculate the second moment of the gradient given the decision vector
    #     :param dec: decision (dim * 1)
    #     :return: second moment of the population utility
    #     """
    #     return (np.linalg.norm(self.grad_utility_ss(dec))**2).item()

    def normalize_state(self, state_mat) -> np.ndarray:
        """
        normalize the state matrix of the population
        :param state_mat: matrix containing the states (i.e., preferences) of the population
        """
        # normalization
        state_norm = np.linalg.norm(state_mat, axis=0, keepdims=True)  # size: 1* column_size
        return self.radi * state_mat / state_norm

    def _initialize_uniform(self) -> np.ndarray:
        """Generate uniformly distributed initial states."""
        # the state can point to any direction
        state_init = np.random.normal(loc=0, scale=2, size=(self.dim, self.size_pop))
        return self.normalize_state(state_init)

    def _initialize_uniform_restricted(self, angle_bd: float) -> np.ndarray:
        """
        Generate uniformly distributed initial states lying in a restricted sphere,
        with angles uniformly distributed within the specified bound
        :param: angle_bd: lower bound on the relative angle
                admissible range: [angle_bd, 180]
        """
        cos_bd = np.cos(np.deg2rad(angle_bd))
        vec_mat = []

        # init_ref = np.zeros((self.dim, 1))  # initial reference vector
        # init_ref[:2, 0] = 1
        init_ref = np.random.normal(loc=0, scale=2, size=(self.dim, 1))
        init_ref /= np.linalg.norm(init_ref)

        while True:
            total_num = sum(batch.shape[1] for batch in vec_mat)
            # break if we generate sufficient amount of data
            if total_num >= self.size_pop:
                break

            # generate more candidate vectors
            vec_cur = np.random.normal(loc=0, scale=2, size=(self.dim, 4 * self.size_pop))
            vec_cur = self.normalize_state(vec_cur)  # normalization
            valid_mask = (init_ref.T @ vec_cur <= self.radi*cos_bd).reshape(-1)  # 1d index array
            vec_valid = vec_cur[:, valid_mask]
            vec_mat.append(vec_valid[:, :max(0, self.size_pop - total_num)])

        return np.hstack(vec_mat)

    def _initialize_bimodal(self) -> np.ndarray:
        """Generate bimodal initial states."""
        state_modal_1 = np.random.normal(-10, 5, size=(self.dim, self.size_pop // 2))
        state_modal_2 = np.random.normal(5, 3, size=(self.dim, self.size_pop // 2))
        state_init = np.hstack((state_modal_1, state_modal_2))
        return self.normalize_state(state_init)

    def _initialize_from_file(self) -> np.ndarray:
        """Load initial states from a saved file."""
        directory = Path(f'Data/dim_{self.dim}')
        subfolders = sorted([f for f in directory.iterdir() if f.is_dir()])
        latest_subfolder = subfolders[-1]
        npz_files = sorted([f for f in latest_subfolder.glob(f'*.npz')])
        state_pop_dict = np.load(npz_files[-1])
        return state_pop_dict['state_pop_init']


# # Test the gradient function from automatic differentiation
# def main():
#     # Initialize the population object
#     dim = 5
#     x0 = np.ones((dim, 1)) / np.sqrt(dim)  # initial guess
#     pop = Population(dim=5, radi=1, size_pop=100, angle_bd=100, sigma=0.5, lambda_p=0.2, dec_init=x0)
#     pop.state_init = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
#     pop.state_cur = pop.state_init
#
#     # Define the decision vector
#     dec = jnp.array([[0.1], [0.2], [0.3]])
#
#     # Calculate the gradient of utility_ss with respect to dec
#     grad_utility_ss = pop.grad_utility_ss(dec)
#     print("Gradient of utility_ss with respect to dec:", grad_utility_ss)
#     sec_moment_dec = pop.sec_moment(dec)
#     print("Second moment of dec given utility_ss:", sec_moment_dec)
#
#     # Calculate the optimal solution
#     x0 = pop.radi * np.ones(pop.dim) / np.sqrt(pop.dim)  # initial guess
#     pop.opt_dec, pop.opt_val = pop.maximize_obj(x0, solver="ipopt")  # use "scipy" or "ipopt" as the solver


# Test the maximization function
def main():
    # Initialize the population object
    np.random.seed(10)
    dim = 5
    x0 = np.ones((dim, 1)) / np.sqrt(dim)  # initial guess
    pop = Population(dim=5, radi=1, size_pop=100, angle_bd=100, sigma=0.5, lambda_p=0.2, dec_init=x0)
    pop.population_init("uniform")

    # pop.state_init = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    # pop.state_cur = pop.state_init
    #
    # # Define the decision vector
    # dec = np.array([[0.1], [0.2], [0.3]])
    #
    # # Calculate the optimal solution
    # x0 = pop.radi * np.ones(pop.dim) / np.sqrt(pop.dim)  # initial guess
    # pop.opt_dec, pop.opt_val = pop.maximize_obj(x0, solver="ipopt")  # use "scipy" or "ipopt" as the solver


if __name__ == "__main__":
    main()

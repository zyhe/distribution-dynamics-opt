"""
Analyze the closed-loop response of the user and the algorithm
We address distribution dynamics involving softmax and an equality constraint on the budget
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import yaml
import cProfile  # for profile analysis
import pstats
import logging
import argparse
from scipy.stats import wasserstein_distance

import warnings
warnings.simplefilter("error", RuntimeWarning)

from Models.distribution_dynamics import UserHedge
from Solvers.vanilla import VanillaAlg
from Solvers.composite import CompositeAlg
from Solvers.vanilla_gf import VanillaGFAlg

# Configure font settings
plt.rcParams.update({
    "font.family": ["Times New Roman", "serif"],  # another option "serif"
    "font.size": 14,
    "mathtext.fontset": "cm",  # Computer Modern for math
})

# Configure logging
logging.basicConfig(
    filename='cl_hedge.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Create a StreamHandler to flush log messages immediately
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# Define a dictionary to map modes to color codes
COLORS_LIST = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#9467bd",  # purple
    3: "#76B7B2"  # muted green
}


class ClosedLoopResponse:
    def __init__(self, file_path: str = 'Config/params.yaml'):
        self.file_path = file_path
        self.params = self._load_parameters()

        # parameters related to the distribution dynamics
        self.lambda1 = self.params['dynamics']['lambda1']
        self.lambda2 = self.params['dynamics']['lambda2']
        self.epsilon = self.params['dynamics']['epsilon']

        # parameters related to the algorithms
        self.sz = self.params['algorithm']['sz']
        self.num_itr = int(float(self.params['algorithm']['num_itr']))
        self.sz_gf = self.params['algorithm']['sz_gf']
        self.delta = self.params['algorithm']['delta']
        self.num_trial = self.params['algorithm']['num_trial']

        # self.penalty_coeff = self.params['algorithm']['penalty_coeff']
        # self.penalty_inc_factor = self.params['algorithm']['penalty_inc_factor']

        # parameters related to the problem
        self.dim = self.params['problem']['dim']
        # self.bd_dec = self.params['problem']['bd_dec']
        self.budget = self.params['problem']['budget']
        self.w_entropy = self.params['problem']['w_entropy']

        # choose the initial guess
        dec_init_unnormalized = np.random.rand(self.dim, 1)
        self.dec_init = self.budget / np.sum(dec_init_unnormalized) * dec_init_unnormalized
        # self.dec_init = self.budget / self.dim * np.ones((self.dim, 1))  # equal weight to each element
        # self.dec_init = self.budget * (self.user.p_init == np.max(self.user.p_init)).astype(int)  # greedy decision

        # initialize the user distribution and the algorithm
        self.user = UserHedge(self.dim, self.lambda1, self.lambda2, self.epsilon, self.budget,
                              self.w_entropy, self.dec_init)
        self.alg = None  # type of the solver
        self.alg_name = {0: 'vanilla', 1: 'composite', 2: 'vanilla_gf'}

        # value array used in calculating Wasserstein distances
        # these values do not matter too much and can be scaled
        ele_val_gap = 1
        self.ele_val = np.arange(0, 0 + self.dim*ele_val_gap, ele_val_gap)

        # store results of the vanilla and composite algorithms
        self.pref_data = np.zeros((2, self.dim, self.num_itr))  # preference state
        self.dec_data = np.zeros((2, self.dim, self.num_itr))  # decision
        self.utility_data = np.zeros((2, self.num_itr))  # utility
        self.constraint_vio_data = np.zeros((2, self.num_itr))  # constraint violation
        self.dist_opt_pt_data = np.zeros((2, self.num_itr))  # distance to the optimal point
        self.dist_wasserstein_ss_own = np.zeros((2, self.num_itr))  # Wasserstein distance to its own steady state
        self.dist_wasserstein_ss_opt = np.zeros((2, self.num_itr))  # Wasserstein distance to the optimal steady state

        # store results corresponding to the gradient-free method
        # the rows represent the lower bound, the mean, and the upper bound of all the trials
        self.utility_data_gf = None
        self.dist_opt_pt_data_gf = None
        self.dist_wasserstein_ss_own_gf = None
        self.dist_wasserstein_ss_opt_gf = None

        # set the path to store results
        self.data_dir = self.setup_dir("Data")
        self.fig_dir = self.setup_dir("Figures")
        self.file_name_dict = {
            "Loss": "loss",
            "Relative optimality gap": "val_gap",
            "Relative distance to $q^*$": "pt_dist",
            r"$W_1(p_k, p_{\text{ss}}(q_k))$": "wass_dist_own",
            r"$W_1(p_k, p_{\text{ss}}(q^{*}))$": "wass_dist_opt"
        }

    def _load_parameters(self) -> dict:
        """
        Load configuration parameters from a YAML file.
        :param file_path: Path to the YAML configuration file.
        :return: Dictionary of loaded parameters.
        """
        with Path(self.file_path).open('r') as file:
            return yaml.safe_load(file)

    def _select_solver(self, mode):
        """Select the solver based on the given mode."""
        if mode == "vanilla":
            return VanillaAlg(self.sz)
        elif mode == "composite":
            return CompositeAlg(self.sz)
        elif mode == "vanilla_gf":
            return VanillaGFAlg(self.sz_gf, self.delta)
        else:
            raise ValueError(f"Unsupported solver type: {mode}")

    def feedback_response(self, index):
        """
        Implement the response when the algorithm is interconnected with the distribution dynamics
        :param index: index of the problem, 0 for vanilla, and 1 for composite
        """
        # initial decision
        dec = self.dec_init
        # penalty_cur = self.penalty_coeff

        for i in range(self.num_itr):
            p_cur = self.user.p_cur
            p_ss = self.user.steady_state(dec)  # steady state

            # calculate the Wasserstein distance relative to the steady-state distribution
            self.dist_wasserstein_ss_own[index, i], self.dist_wasserstein_ss_opt[index, i] \
                = self._compute_wasserstein_distances(p_cur, p_ss, self.user.opt_ss)

            # store results
            self.pref_data[index, :, i:i+1] = p_cur
            self.dec_data[index, :, i:i+1] = dec
            self.utility_data[index, i], self.constraint_vio_data[index, i] = self.evaluate_perf(dec, p_ss)

            # penalty_cur *= self.penalty_inc_factor
            _ = self.user.per_step_dynamics(dec)  # distribution dynamics
            dec = self.alg.itr_update(dec, self.user, self.budget)  # algorithmic update

        # calculate the distance to the optimal point
        self.dist_opt_pt_data[index] = np.linalg.norm(self.dec_data[index] - self.user.opt_pt, axis=0)

        logging.info(f'The {self.alg_name[index]} algorithm is finished.')
        logging.info(f'The decision vector is {self.dec_data[index, :, -1].T}.')
        logging.info(f'The final objective value is {-self.user.objective_function(self.dec_data[index, :, -1])}.')

        # reset the user profile
        self.user.reset()

    def feedback_response_gf(self):
        """
        Implement the response when the gradient-free algorithm is applied to the distribution dynamics
        """
        # store results of all trials
        utility_data_gf_all = np.zeros((self.num_trial, self.num_itr))
        dist_opt_pt_data_gf_all = np.zeros((self.num_trial, self.num_itr))
        dist_wasserstein_ss_own_gf_all = np.zeros((self.num_trial, self.num_itr))
        dist_wasserstein_ss_opt_gf_all = np.zeros((self.num_trial, self.num_itr))

        for trial in range(self.num_trial):
            # initial decision
            dec = self.dec_init
            dec_data_gf_trial = np.zeros((self.dim, self.num_itr))

            for i in range(self.num_itr):
                p_cur = self.user.p_cur
                p_ss = self.user.steady_state(dec)  # steady state

                # calculate the Wasserstein distance relative to the steady-state distribution
                dist_wasserstein_ss_own_gf_all[trial, i], dist_wasserstein_ss_opt_gf_all[trial, i] = (
                    self._compute_wasserstein_distances(p_cur, p_ss, self.user.opt_ss))
                # store results
                dec_data_gf_trial[:, i:i+1] = dec
                utility_data_gf_all[trial, i], _ = self.evaluate_perf(dec, p_ss)

                # sample a vector uniformly from the unit sphere
                v = np.random.randn(self.dim, 1)
                v /= np.linalg.norm(v)  # normalization
                # perturb the decision
                dec_pert = dec + self.delta * v

                _ = self.user.per_step_dynamics(dec_pert)  # distribution dynamics
                dec = self.alg.itr_update(dec, v, self.user, self.budget)  # algorithmic update

            # calculate the distance to the optimal point
            dist_opt_pt_data_gf_all[trial, :] = np.linalg.norm(dec_data_gf_trial - self.user.opt_pt, axis=0)
            # reset the user profile after each trial
            self.user.reset()
            logging.info(f'The {trial}-th trial of the vanilla gradient-free algorithm is finished.')

        self.utility_data_gf = self.process_trial_data(utility_data_gf_all)
        self.dist_opt_pt_data_gf = self.process_trial_data(dist_opt_pt_data_gf_all)
        self.dist_wasserstein_ss_own_gf = self.process_trial_data(dist_wasserstein_ss_own_gf_all)
        self.dist_wasserstein_ss_opt_gf = self.process_trial_data(dist_wasserstein_ss_opt_gf_all)

        logging.info(f'The vanilla gradient-free algorithm is finished.')
        logging.info(f'The decision vector is {dec_data_gf_trial[:, -1].T}.')
        logging.info(f'The final objective value is {-self.user.objective_function(dec_data_gf_trial[:, -1])}.')

    def _compute_wasserstein_distances(self, p_cur, p_ss, p_opt):
        """
        Helper method to compute Wasserstein distances.
        :param p_cur: current preference distribution vector
        :param p_ss: current steady state distribution vector corresponding to the decision
        :param p_opt: optimal preference distribution vector
        :return Wasserstein distances between p_cur and p_ss, as well as p_cur and p_opt
        """
        return (
            wasserstein_distance(self.ele_val, self.ele_val, p_cur.flatten(), p_ss.flatten()),
            wasserstein_distance(self.ele_val, self.ele_val, p_cur.flatten(), p_opt.flatten())
        )

    def evaluate_perf(self, dec: np.ndarray, p_ss: np.ndarray) -> tuple[float, float]:
        """
        Evaluate the performance in terms of the objective and constraint satisfaction.
        :param dec: decision vector
        :param p_ss: steady-state distribution vector corresponding to the decision
        :return: values of utility and constraint violation
        """
        utility = (p_ss.T @ dec + self.w_entropy * np.sum(p_ss * np.log(p_ss))).item()
        constraint_violation = (np.sum(dec) - self.budget)**2
        return utility, constraint_violation

    def visualize(self):
        """Plot utility and constraint violation over iterations."""
        # self._plot_metric(self.utility_data, "Loss")  # utility
        # self._plot_metric(self.constraint_vio_data, "Constraint Violation")  # constraint violation

        self._semilog_metric((self.user.opt_val - self.utility_data)/self.user.opt_val,
                             (self.user.opt_val - self.utility_data_gf)/self.user.opt_val,
                             "Relative optimality gap")  # gap relative to the optimal utility
        self._semilog_metric(self.dist_opt_pt_data/np.linalg.norm(self.user.opt_pt),
                             self.dist_opt_pt_data_gf/np.linalg.norm(self.user.opt_pt),
                             "Relative distance to $q^*$")  # gap relative to the optimal utility
        # self._semilog_metric(self.dist_wasserstein_ss_own, self.dist_wasserstein_ss_own_gf,
        #                      r"$W_1(p_k, p_{\text{ss}}(q_k))$")  # Wasserstein distance
        self._semilog_metric(self.dist_wasserstein_ss_opt, self.dist_wasserstein_ss_opt_gf,
                             r"$W_1(p_k, p_{\text{ss}}(q^{*}))$")  # Wasserstein distance
        plt.show()

    def _plot_metric(self, data: np.ndarray, ylabel: str):
        """Plot the evolution of a specific metric."""
        plt.figure()
        plt.plot(np.arange(self.num_itr), data[0], linewidth=3, label='vanilla', color=COLORS_LIST[0])
        plt.plot(np.arange(self.num_itr), data[1], linewidth=3, label='composite', color=COLORS_LIST[1])
        plt.legend(fontsize=16, loc='upper right')
        plt.xlim([0, self.num_itr])
        plt.xlabel('Number of iterations', fontsize=18)
        plt.ylabel(ylabel, fontsize=18, usetex=False)
        plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=12, direction='in', length=2, width=0.5)  # Minor ticks
        plt.minorticks_on()
        plt.locator_params(axis='both', nbins=7)
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.fig_dir / f'{self.file_name_dict[ylabel]}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show(block=False)

    def _semilog_metric(self, data: np.ndarray, data_gf: np.ndarray, ylabel: str):
        """
        Use the semilog plot.
        :param data: results of the vanilla and composite algorithm
        :param data_gf: results of the vanilla gradient-free algorithm
        :param ylabel: ylabel
        """
        plt.figure()
        plt.semilogy(np.arange(self.num_itr), data[0], linewidth=3, label='vanilla', color=COLORS_LIST[0])
        plt.semilogy(np.arange(self.num_itr), data[1], linewidth=3, label='composite', color=COLORS_LIST[1])
        plt.fill_between(np.arange(self.num_itr), data_gf[0], data_gf[2], alpha=0.3,
                         color=COLORS_LIST[2], edgecolors=None)
        plt.semilogy(np.arange(self.num_itr), data_gf[1], linewidth=2, label='derivative-free', color=COLORS_LIST[2])
        plt.legend(fontsize=16, loc='upper right')
        plt.xlim([0, self.num_itr])
        plt.xlabel('Number of iterations', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=12, direction='in', length=2, width=0.5)  # Minor ticks
        plt.minorticks_on()
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.fig_dir / f'sz-{self.sz_gf}-delta-{self.delta}-{self.file_name_dict[ylabel]}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.savefig(self.fig_dir / f'{self.file_name_dict[ylabel]}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show(block=False)

    def save_results(self):
        """Save results to NPZ files with a timestamped directory."""
        # save convergence results
        np.savez(
            self.data_dir / "conv_data.npz",
            opt_val=self.user.opt_val,
            opt_pt=self.user.opt_pt,
            utility_data=self.utility_data,
            utility_data_gf=self.utility_data_gf,
            dist_opt_pt_data=self.dist_opt_pt_data,
            dist_opt_pt_data_gf=self.dist_opt_pt_data_gf,
            dist_wasserstein_ss_own=self.dist_wasserstein_ss_own,
            dist_wasserstein_ss_own_gf=self.dist_wasserstein_ss_own_gf,
            dist_wasserstein_ss_opt=self.dist_wasserstein_ss_opt,
            dist_wasserstein_ss_opt_gf=self.dist_wasserstein_ss_opt_gf
        )
        logging.info("Results saved successfully.")

    def setup_dir(self, typ_name: str):
        """
        Set up the directory for storing results, e.g., data and figures
        :param typ_name: type, either "Data" or "Figures"
        :return target_dir: target directory for storing results
        """
        # obtain time information
        current_time = datetime.now()
        time_suffix = current_time.strftime('%Y-%m-%d_%H-%M-%S')

        # specify the directory
        target_dir = Path(f'./{typ_name}/{time_suffix}-sz-{self.sz_gf}-delta-{self.delta}')
        # target_dir = Path(f'./{typ_name}')
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def execute(self):
        """
        Executes full closed-loop responses with different algorithms.
        """
        # Run the vanilla algorithm
        self.alg = self._select_solver("vanilla")
        self.feedback_response(index=0)

        # Run the composite algorithm
        self.alg = self._select_solver("composite")
        self.feedback_response(index=1)

        # Run the vanilla gradient-free algorithm
        self.alg = self._select_solver("vanilla_gf")
        self.feedback_response_gf()

        self.visualize()
        self.save_results()
        logging.info('Finish the program.')

    @staticmethod
    def profile_execution():
        """
        Profiles the execution of the main analysis process.
        """
        cProfile.run("ClosedLoopResponse().execute()", "main_stats")
        p = pstats.Stats("main_stats")
        p.sort_stats("cumulative").print_stats(50)

    @staticmethod
    def process_trial_data(trial_data: np.ndarray) -> np.ndarray:
        """
        Process the trial data and return an array containing the maximum, mean, and minimum of each column
        :param trial_data: num_trial-by-num_itr
        :return: condensed_data: 3-by-num_itr
        """
        num_itr = trial_data.shape[1]
        condensed_data = np.zeros((3, num_itr))
        condensed_data[0] = np.min(trial_data, axis=0)
        condensed_data[1] = np.mean(trial_data, axis=0)
        condensed_data[2] = np.max(trial_data, axis=0)
        return condensed_data


def main():
    np.random.seed(10)

    parser = argparse.ArgumentParser(description="Analyze the closed-loop response of the system.")
    parser.add_argument("index", nargs='?', type=int, default=None,
                        help="Optional index to select a specific params file.")
    args = parser.parse_args()

    if args.index is None:
        file_path = 'Config/params_3.yaml'
    else:
        file_path = f'Config/params_{args.index}.yaml'

    response_runner = ClosedLoopResponse(file_path=file_path)
    response_runner.execute()
    # response_runner.profile_execution()


if __name__ == "__main__":
    main()

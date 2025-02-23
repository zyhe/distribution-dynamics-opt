"""
Analyze the closed-loop response of the system
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import yaml
import cProfile  # for profile analysis
import pstats
import logging
import argparse
# import pickle

from Models.cl_response import CLResponse, construct_mode_dict

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
    3: "#2ca02c"  # green
}


class ClosedLoopRunner:
    def __init__(self, file_path: str = 'Config/params.yaml'):
        self.file_path = file_path
        self.params = self._load_parameters()
        # parameters related to the population
        self.dim = self.params['population']['dim']
        self.radi = self.params['population']['radi']
        self.size_pop = self.params['population']['size_pop']
        self.angle_bd = self.params['population']['angle_bd']
        self.sigma = self.params['population']['sigma']
        self.lambda_p = self.params['population']['lambda_p']
        self.distri_md = self.params['population']['distri_md']

        # parameters related to the algorithm
        self.sz = self.params['algorithm']['sz']
        self.num_sample = self.params['algorithm']['num_sample']
        self.num_itr = int(float(self.params['algorithm']['num_itr']))
        self.num_trial = int(float(self.params['algorithm']['num_trial']))

        # allocate results arrays
        self.uti_dynamic_result = np.zeros((2, self.num_trial, self.num_itr))  # utility
        self.uti_ss_result = np.zeros((2, self.num_trial, self.num_itr))  # utility
        self.dist_pt_result = np.zeros((2, self.num_trial, self.num_itr))  # distance to the optimal point
        self.state_pop_final_overall = np.zeros((3, self.dim, self.size_pop))  # final state
        # order of the rows in state_pop_final_overall : vanilla, composite, optimal
        self.angle_data_overall = {
            'alg': np.zeros((2, self.num_trial * self.size_pop)),
            'optimal': np.zeros(self.size_pop),
            'initial': np.zeros(self.size_pop)
        }  # angle; order of the rows in the value of "alg": vanilla, composite

        # specify the directory
        self.data_dir = self.setup_dir("Data")
        self.fig_dir = self.setup_dir("Figures")

        # initialize the closed-loop response object
        self.cl = CLResponse(dim=self.dim, radi=self.radi, size_pop=self.size_pop, angle_bd=self.angle_bd,
                             sigma=self.sigma, lambda_p=self.lambda_p, num_sample=self.num_sample,
                             sz=self.sz, num_itr=self.num_itr, num_trial=self.num_trial,
                             mode_dict=construct_mode_dict("vanilla", self.distri_md))

    def _load_parameters(self) -> dict:
        """
        Load configuration parameters from a YAML file.
        :param file_path: Path to the YAML configuration file.
        :return: Dictionary of loaded parameters.
        """
        with Path(self.file_path).open('r') as file:
            return yaml.safe_load(file)

    def save_results(self):
        """Save results to CSV and NPZ files with a timestamped directory."""
        # # create a DataFrame
        # data = {
        #     'Itr': np.arange(0, self.num_itr),
        #     'utli_vanilla': self.uti_dynamic_result[0],
        #     'utli_composite': self.uti_dynamic_result[1]
        # }
        # conv_result = pd.DataFrame(data)
        # # save convergence results to CSV
        # conv_result.to_csv(self.data_dir / f'conv.csv', sep=',', index=False)

        # save convergence results
        np.savez(
            self.data_dir / "conv_data.npz",
            uti_ss_result=self.uti_ss_result,
            dist_pt_result=self.dist_pt_result,
            opt_dec=self.cl.pop.opt_dec,
            opt_val=self.cl.pop.opt_val
        )

        # save the data of the states of the population
        np.savez(
            self.data_dir / "state_angle_data.npz",
            state_pop_init=self.cl.pop.state_init,
            state_pop_final_overall=self.state_pop_final_overall,
            angle_data_overall=self.angle_data_overall,
            hist_idx=self.cl.hist_idx
        )

        logging.info("Results saved successfully.")

        # # save the object
        # with open('Data/cl_obj_{}.pkl'.format(time_suffix), 'wb') as file:
        #     pickle.dump(cl, file, pickle.HIGHEST_PROTOCOL)

    def plot_conv_measure(self):
        """Plot utility curves"""
        self.plot_convergence_curves(self.uti_ss_result, self.num_itr, ["Vanilla", "Composite"],
                                     "Utility", log_scale=False)

    def semilog_conv_measure(self):
        """Plot convergence measures in a semilog graph"""
        # relative optimality gap
        relative_gap = (self.cl.pop.opt_val - self.uti_ss_result) / self.cl.pop.opt_val  # self.uti_dynamic_result
        self.plot_convergence_curves(relative_gap, self.num_itr, ["Vanilla", "Composite"],
                                     "Relative optimality gap", log_scale=True)

        # relative distance to the optimal point
        relative_dist = self.dist_pt_result / np.linalg.norm(self.cl.pop.opt_dec)
        self.plot_convergence_curves(relative_dist, self.num_itr, ["Vanilla", "Composite"],
                                     "Relative distance", log_scale=True)  # to $q^*

    def plot_convergence_curves(self, data: np.ndarray, num_itr: int, labels: list,
                                ylabel: str, log_scale: bool = False):
        """
        General function to plot convergence curves.
        :param data: numpy data arrays.
        :param num_itr: number of iterations.
        :param labels: list of labels for the plots.
        :param ylabel: y-axis label.
        :param log_scale: whether to log scale.
        """
        plt.figure()
        x = range(num_itr)
        for idx, label in enumerate(labels):
            low = np.min(data[idx], axis=0)
            high = np.max(data[idx], axis=0)
            mean = np.mean(data[idx], axis=0)
            color = COLORS_LIST[idx]  # access the color
            if log_scale:
                plt.fill_between(x, low, high, alpha=0.3, label=label, color=color, edgecolors=None)
                plt.semilogy(x, mean, linewidth=2, color=color)
            else:
                plt.fill_between(x, low, high, alpha=0.3, label=label, color=color, edgecolors=None)
                plt.plot(x, mean, linewidth=2, color=color)

        plt.legend(fontsize=14)
        plt.xlabel('Number of Iterations', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=10, direction='in', length=2, width=0.5)
        plt.minorticks_on()
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.fig_dir / f'{ylabel}-bd-{self.angle_bd}-lambda-{self.lambda_p}-sigma-{self.sigma}-sz-{self.sz}.png',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show()

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
        target_dir = Path(
            f'./{typ_name}/{time_suffix}-bd-{self.angle_bd}-lambda-{self.lambda_p}-sigma-{self.sigma}-sz-{self.sz}')
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def run_algorithm(self, cl: CLResponse, solver: str, distribution: str, index: int) -> None:
        """
        Run the specified algorithm and store results.
        :param cl: Initialized CLResponse object.
        :param solver: Solver mode ("vanilla" or "composite").
        :param distribution: Data distribution mode ("uniform", "bimodal", or "load").
        :param index: Index for storing results in arrays. 0: vanilla, 1: composite.
        """
        mode_dict = construct_mode_dict(solver, distribution)
        self.cl.mode_dict = mode_dict
        self.cl.response()
        self.uti_dynamic_result[index] = self.cl.utility_dynamic_data
        self.uti_ss_result[index] = self.cl.utility_ss_data
        self.dist_pt_result[index] = self.cl.dist_pt_data
        self.state_pop_final_overall[index] = self.cl.pop.state_final
        self.angle_data_overall['alg'][index] = self.cl.pop.angle_final

    def execute(self):
        """
        Executes full closed-loop responses with different algorithms.
        """
        # Evaluate the results generated by the optimal decision
        self.cl.response_opt_dec()
        self.state_pop_final_overall[2] = self.cl.pop.state_final
        self.angle_data_overall['optimal'] = self.cl.pop.angle_final
        self.angle_data_overall['initial'] = self.cl.pop.angle_init
        # self.angle_data_overall[2:, :] = np.vstack((self.cl.pop.angle_final, self.cl.pop.angle_init))

        # Run the vanilla algorithm
        self.run_algorithm(self.cl, "vanilla", self.distri_md, index=0)
        # Run the composite algorithm
        self.run_algorithm(self.cl, "composite", self.distri_md, index=1)

        self.plot_conv_measure()
        self.semilog_conv_measure()
        self.save_results()
        plt.show()

        return None
        # print('Finish the program')


def profile_execution(file_path):
    """
    Profile the execution of the main analysis process.
    """
    cProfile.run(f"ClosedLoopRunner('{file_path}').execute()", "main_stats")
    p = pstats.Stats("main_stats")
    p.sort_stats("cumulative").print_stats(50)


def main():
    # set the random seed
    # mind 4, 6, 15, 16, 17, 200
    np.random.seed(20)

    parser = argparse.ArgumentParser(description="Analyze the closed-loop response of the system.")
    parser.add_argument("index", nargs='?', type=int, default=None,
                        help="Optional index to select a specific params file.")
    args = parser.parse_args()

    if args.index is None:
        file_path = 'Config/params_3.yaml'
    else:
        file_path = f'Config/params_{args.index}.yaml'

    response_runner = ClosedLoopRunner(file_path=file_path)
    response_runner.execute()
    
    # If profile test is desired, then comment the above two lines and uncomment the following line
    # profile_execution(file_path)


if __name__ == '__main__':
    main()

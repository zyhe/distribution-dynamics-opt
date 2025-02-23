"""
## plot convergence results and histograms of the population
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure font settings
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "mathtext.fontset": "cm",  # Computer Modern for math
})

# Define a dictionary to map modes to color codes
COLORS_LIST = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#9467bd",  # purple
    3: "#76B7B2"  # muted green
}

MODES_DICT = {
    0: "vanilla",
    1: "composite",
    2: "optimal"
}


class ConvergencePlotter:
    def __init__(self, idx: int) -> None:
        self.data_dir = Path(f'../Data')
        self.cur_data_subfolder = self._get_subfolder(idx)
        self.conv_file = self.cur_data_subfolder / 'conv_data.npz'

        # Load NPZ data
        self.conv_dict = dict(np.load(self.conv_file))

        # Specify and create figure directory
        self.fig_dir = Path(__file__).parent / self.cur_data_subfolder.name
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Collect related data
        self.opt_val = self.conv_dict['opt_val']
        self.utility_data = self.conv_dict['utility_data']
        self.utility_data_gf = self.conv_dict['utility_data_gf']
        self.dist_opt_pt_data = self.conv_dict['dist_opt_pt_data']
        self.dist_opt_pt_data_gf = self.conv_dict['dist_opt_pt_data_gf']
        # self.dist_wasserstein_ss_own = self.conv_dict['dist_wasserstein_ss_own']
        # self.dist_wasserstein_ss_own_gf = self.conv_dict['dist_wasserstein_ss_own_gf']
        self.dist_wasserstein_ss_opt = self.conv_dict['dist_wasserstein_ss_opt']
        self.dist_wasserstein_ss_opt_gf = self.conv_dict['dist_wasserstein_ss_opt_gf']

        self.file_name_dict = {
            "Loss": "loss",
            "Optimality gap": "val_gap",
            "Distance to $q^*$": "pt_dist",
            r"$W_1(p_k, p_{\text{ss}}(q_k))$": "wass_dist_own",
            r"$W_1(p_k, p_{\text{ss}}(q^{*}))$": "wass_dist_opt"
        }

    def _get_subfolder(self, idx: int) -> Path:
        """Get the idx-th subfolder from the specified directory."""
        subfolders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()], reverse=True)
        return subfolders[idx] if subfolders else None

    def semilog_conv_measure(self):
        """Plot convergence measures in a semilog graph"""
        self._semilog_metric(self.opt_val-self.utility_data, self.opt_val-self.utility_data_gf,
                             "Optimality gap", ylim=1e-6)  # gap relative to the optimal utility
        self._semilog_metric(self.dist_opt_pt_data, self.dist_opt_pt_data_gf,
                             "Distance to $q^*$", ylim=1e-4)  # gap relative to the optimal utility
        # self._semilog_metric(self.dist_wasserstein_ss_own, self.dist_wasserstein_ss_own_gf,
        #                      r"$W_1(p_k, p_{\text{ss}}(q_k))$")  # Wasserstein distance
        self._semilog_metric(self.dist_wasserstein_ss_opt, self.dist_wasserstein_ss_opt_gf,
                             r"$W_1(p_k, p_{\text{ss}}(q^{*}))$", ylim=1e-5) # Wasserstein distance
        # plt.show()

    def _semilog_metric(self, data: np.ndarray, data_gf: np.ndarray, ylabel: str, ylim: float = 0):
        """
        Use the semilog plot.
        :param data: results of the vanilla and composite algorithm
        :param data_gf: results of the vanilla gradient-free algorithm
        :param ylabel: ylabel
        """
        num_itr = data.shape[-1]
        plt.figure()
        plt.semilogy(np.arange(num_itr), data[0], linewidth=3, label='vanilla', color=COLORS_LIST[0])
        plt.semilogy(np.arange(num_itr), data[1], linewidth=3, label='composite', color=COLORS_LIST[1])
        plt.fill_between(np.arange(num_itr), data_gf[0], data_gf[2], alpha=0.3,
                         color=COLORS_LIST[2], edgecolors=None)
        plt.semilogy(np.arange(num_itr), data_gf[1], linewidth=2, label='derivative-free', color=COLORS_LIST[2])
        plt.legend(fontsize=16, loc='upper right')
        plt.xlim([0, 5000])
        if ylim != 0:
            plt.ylim(bottom=ylim)
        plt.xlabel('Number of iterations', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=12, direction='in', length=2, width=0.5)  # Minor ticks
        plt.minorticks_on()
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.fig_dir / f'{self.file_name_dict[ylabel]}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    def run_all(self):
        """Show all the results."""
        # Plot convergence
        self.semilog_conv_measure()
        print(f'Subfolder {self.cur_data_subfolder.name} is successfully processed.')
        pass


def main():
    # Ensure working directory is set to script's location
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    num_subfolders = 4  # number of subfolders to be loaded
    for idx in range(num_subfolders):
        plotter = ConvergencePlotter(idx)
        plotter.run_all()


if __name__ == '__main__':
    main()
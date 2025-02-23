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

ORDINAL_DICT = {
    0: "0th", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
    6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
    11: "11th", 12: "12th", 13: "13th", 14: "14th",
    15: "15th", 16: "16th", 17: "17th", 18: "18th",
    19: "19th", 20: "20th"
}

# Define a dictionary to map modes to color codes
COLORS_LIST = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#ff7f0e",  # orange
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
        self.state_angle_file = self.cur_data_subfolder / 'state_angle_data.npz'

        # Load NPZ data
        self.conv_dict = dict(np.load(self.conv_file))
        self.state_angle_dict = dict(np.load(self.state_angle_file, allow_pickle=True))

        # Specify and create figure directory
        self.fig_dir = Path(__file__).parent / self.cur_data_subfolder.name
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.hist_dir = self.fig_dir / 'Histograms'
        self.hist_dir.mkdir(parents=True, exist_ok=True)

        # Collect related data
        self.uti_ss_result = self.conv_dict['uti_ss_result']
        self.dist_pt_result = self.conv_dict['dist_pt_result']
        self.num_itr = self.uti_ss_result.shape[-1]
        self.opt_dec = self.conv_dict['opt_dec']
        self.opt_val = self.conv_dict['opt_val']

        self.state_init = self.state_angle_dict['state_pop_init']
        self.state_pop_final_overall = self.state_angle_dict['state_pop_final_overall']
        self.angle_data_overall = self.state_angle_dict['angle_data_overall'].item()

        self.hist_idx = self.state_angle_dict['hist_idx']

        self.conv_file_name_dict = {
            "Relative optimality gap": "val_gap",
            "Relative distance to $q^*$": "pt_dist"
        }

    def _get_subfolder(self, idx: int) -> Path:
        """Get the idx-th subfolder from the specified directory."""
        subfolders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()], reverse=True)
        return subfolders[idx] if subfolders else None

    # def plot_conv_measure(self):
    #     """Plot utility curves"""
    #     uti_ss_result = self.conv_dict['uti_ss_result']
    #     num_itr = uti_ss_result.shape[-1]
    #     self.helper_convergence_curves(uti_ss_result, num_itr, ["Vanilla", "Composite"],
    #                                  "Utility", log_scale=False)

    def semilog_conv_measure(self):
        """Plot convergence measures in a semilog graph"""
        # relative optimality gap
        relative_gap = (self.opt_val - self.uti_ss_result) / self.opt_val
        self.helper_convergence_curves(relative_gap, self.num_itr, ["Vanilla", "Composite"],
                                     "Relative optimality gap", ylim=1e-4, log_scale=True)

        # relative distance to the optimal point
        relative_dist = self.dist_pt_result / np.linalg.norm(self.opt_dec)
        self.helper_convergence_curves(relative_dist, self.num_itr, ["Vanilla", "Composite"],
                                     "Relative distance to $q^*$", ylim=1e-2, log_scale=True)  # to $q^*$

    def helper_convergence_curves(self, data: np.ndarray, num_itr: int, labels: list,
                                  ylabel: str, ylim: float = 0, log_scale: bool = True):
        """
        General function to plot convergence curves.
        :param data: numpy data arrays (size: 2 * num_trial * actual num_itr)
        :param num_itr: number of iterations.
        :param labels: list of labels for the plots.
        :param ylabel: y-axis label.
        :param ylimit: limit of the y-axis.
        :param log_scale: whether to log scale.
        """
        plt.figure()
        x = range(num_itr)
        for idx, label in enumerate(labels):
            plot_data = data[idx, :, 0: num_itr]
            low = np.min(plot_data, axis=0)
            high = np.max(plot_data, axis=0)
            mean = np.mean(plot_data, axis=0)
            color = COLORS_LIST[idx]  # access the color
            if log_scale:
                plt.fill_between(x, low, high, alpha=0.3, label=label, color=color, edgecolors=None)
                plt.semilogy(x, mean, linewidth=2, color=color)
            else:
                plt.fill_between(x, low, high, alpha=0.3, label=label, color=color, edgecolors=None)
                plt.plot(x, mean, linewidth=2, color=color)

        plt.legend(fontsize=14)
        plt.xlim([0, 6000])
        if ylim != 0:
            plt.ylim(bottom=ylim)
        plt.xlabel('Number of iterations')
        plt.ylabel(ylabel)
        plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4, width=0.5)
        plt.tick_params(axis='both', which='minor', labelsize=12, direction='in', length=2, width=0.5)
        plt.minorticks_on()
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.fig_dir / f'{self.conv_file_name_dict[ylabel]}.pdf',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    def position_histogram(self):
        """
        Plot the histograms of positions
        """
        for idx in range(3):
            for pos_id in self.hist_idx:
                self.helper_histogram(idx, pos_id)

    def helper_histogram(self, idx: int, pos_id: int):
        """
        Helper function for plotting the histograms of preferences
        :param idx: index of the mode, i.e., 0-"vanilla", 1-"composite", or 2-"optimal"
        :param pos_id: the index of the element of the preference vector
        """
        mode = MODES_DICT[idx]
        plt.figure()
        plt.hist(self.state_init[pos_id, :], alpha=0.4, label='Initial',
                 # weights=np.ones_like(self.state_init[pos_id, :]) / len(self.state_init[pos_id, :]),
                 color=COLORS_LIST[3], bins=50, density=True)  # initial
        # reason of using np.ones_like: give equal weights to samples
        # bins = 30
        plt.hist(self.state_pop_final_overall[idx, pos_id, :], alpha=0.6, label='Final',
                 # weights=np.ones_like(self.state_init[pos_id, :]) / len(self.state_init[pos_id, :]),
                 color=COLORS_LIST[idx], bins=50, density=True)  # current
        plt.legend()
        plt.xlabel(f'Position ({ORDINAL_DICT[pos_id + 1]} element)')
        plt.ylabel('Probability density')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.title(f'Distribution of preferences ({mode})')
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.hist_dir / f'pos_histogram_{mode}_{pos_id}.pdf',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    def angle_histogram(self):
        """
        Plot the histograms of angles between preferences and the decision
        """
        for idx in range(3):
            self.helper_angle_histogram(idx)

    def helper_angle_histogram(self, idx: int):
        """
        Helper function for plotting the histogram of angles between preferences and the decision
        :param idx: index of the mode, i.e., 0-"vanilla", 1-"composite", or 2-"optimal"
        """
        mode = MODES_DICT[idx]
        angle_init = self.angle_data_overall['initial']

        if idx == 0 or idx == 1:
            angle_final = self.angle_data_overall['alg'][idx]
        else:
            angle_final = self.angle_data_overall['optimal']

        plt.figure()
        plt.hist(angle_init, alpha=0.4, label='Initial',
                 # weights=np.ones_like(angle_init) / len(angle_init),
                 color=COLORS_LIST[3], bins=80, density=True)  # initial
        # bins = 30
        plt.hist(angle_final, alpha=0.6, label='Final',
                 # weights=np.ones_like(angle_final) / len(angle_final),
                 color=COLORS_LIST[idx], bins=80, density=True)  # current
        plt.legend()
        plt.xlabel('Angle ($^{\circ}$)')
        plt.ylabel('Probability density')
        plt.ylim([0, 0.045])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.title(f'Distribution of preference-decision angles ({mode})')
        plt.grid(linestyle='--', color='gray')
        plt.tight_layout(pad=0)
        plt.savefig(self.hist_dir / f'agl_histogram_{mode}.pdf',
                    bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show(block=False)

    def run_all(self):
        """Show all the results."""
        # Plot convergence
        self.semilog_conv_measure()
        
        # Plot histograms
        self.position_histogram()

        # Plot the histogram of angles
        self.angle_histogram()
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

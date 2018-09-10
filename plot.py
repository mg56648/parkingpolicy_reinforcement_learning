import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    """Plot helps keep track of data points and generate plots.
    """
    @staticmethod
    def plot(x, y, title, output_file):
        """Plot helper. Saves plot to output file.

        Args:
            x: list of numbers on x-axis
            y: list of numbers on y-axis
            title: plot title
            output_file: output file
        """
        # plot
        plt.plot(x, y, color='blue', linewidth=2.5, linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Average Total Reward')
        plt.title(title)

        # save figure
        plt.savefig(output_file, dpi=72)
        plt.clf()

    @staticmethod
    def plot_multiple(x_list, y_list, legend_list, legend_prefix, title, output_file):
        """Plot helper for multiple plots. Saves plot to output file.

        Args:
            x: list of list of numbers on x-axis
            y: list of list of numbers on y-axis
            legend_list: list of strings
            legend_prefix: string
            title: plot title
            output_file: output file
        """
        # plot
        for x, y, l in zip(x_list, y_list, legend_list):
            plt.plot(x, y, linewidth=2.5, linestyle='-', label='{0}={1}'.format(legend_prefix, l))
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('Epoch')
        plt.ylabel('Average Total Reward')
        plt.title(title)

        # save figure
        plt.savefig(output_file, dpi=72)
        plt.clf()

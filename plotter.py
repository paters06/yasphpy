from matplotlib import pyplot as plt


class Plotter:
    def __init__():
        pass

    def __init__(self, xlabel_str, ylabel_str):
        self.xlabel_str = xlabel_str
        self.ylabel_str = ylabel_str

    def plot_scatter(self, title_str, points, field):
        plt.figure()
        plt.title(title_str)
        plt.xlabel(self.xlabel_str)
        plt.ylabel(self.ylabel_str)
        plt.scatter(points[0, :], points[1, :], c=field)
        cbar = plt.colorbar()
        cbar.set_label('Temperature °C')

    def show_plots(self, show):
        if show:
            plt.show()

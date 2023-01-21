from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np


def progress_callback(current_frame: int, total_frames: int):
    print(f'Saving frame {current_frame}/{total_frames}')


class Plotter:
    def __init__(self, xlabel_str: str, ylabel_str: str) -> None:
        """
        Parameters
        ----------

            xlabel_str (`str`): title in the x axis
            ylabel_str (`str`): title in the y axis

        Returns
        -------
        `None`

        """
        self.xlabel_str = xlabel_str
        self.ylabel_str = ylabel_str

    def set_image_directory(self, dirname: str):
        """
        Params
        --------
            dirname (`str`): directory where images will be exported if needed
        """
        self.dirname = dirname

    def plot_points(self, title_str: str, points: np.ndarray,
                    pt2: np.ndarray):
        """
        Params
        --------
            title_str (`str`): title of the whole plot
            cbar_str (`str`): title of the colorbar
            points (`np.ndarray`): positions of the scatter points
            field (`np.ndarray`): field values of the scatter points
        """
        x_min = np.min(pt2[:, 0])
        x_max = np.max(pt2[:, 1])
        fig, ax = plt.subplots()
        ax.set_title(title_str)
        ax.set_xlabel(self.xlabel_str)
        ax.set_ylabel(self.ylabel_str)
        ax.set_xlim(-0.5 - x_min, x_max + 0.5)
        im = ax.plot(points[:, 0], points[:, 1], '.k', label='Domain')
        im2 = ax.plot(pt2[:, 0], pt2[:, 1], '.b', label='Virtual')
        ax.legend(loc='right')
        ax.set_aspect('equal')
        return fig

    def plot_scalar_field(self, title_str: str, cbar_str: str,
                          points: np.ndarray, field: np.ndarray):
        """
        Params
        --------
            title_str (`str`): title of the whole plot
            cbar_str (`str`): title of the colorbar
            points (`np.ndarray`): positions of the scatter points
            field (`np.ndarray`): field values of the scatter points
        """
        fig, ax = plt.subplots()
        ax.set_title(title_str)
        ax.set_xlabel(self.xlabel_str)
        ax.set_ylabel(self.ylabel_str)
        im = ax.scatter(points[:, 0], points[:, 1], c=field)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_str)
        ax.set_aspect('equal')
        return fig

    def plot_vector_field(self, title_str: str, cbar_str: str,
                          points: np.ndarray, vec_field: np.ndarray):
        fig, ax = plt.subplots()
        ax.set_title(title_str)
        ax.set_xlabel(self.xlabel_str)
        ax.set_ylabel(self.ylabel_str)
        # mag_field = np.sqrt(vec_field[:, 0]**2 + vec_field[:, 1]**2)
        im = ax.quiver(points[:, 0], points[:, 1],
                       vec_field[:, 0], vec_field[:, 1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_str)
        ax.set_aspect('equal')
        return fig

    def show_plots(self, show):
        if show:
            plt.show()

    @staticmethod
    def save_plot(dirname, fig, filename) -> None:
        fig.savefig(dirname + filename, bbox_inches='tight')

    def make_transient_plot(self, points: np.ndarray, T_transient: np.ndarray,
                            dirname: str, filename: str,
                            title_str: str) -> None:
        """
        Due to implementation issues, when class `ArtistAnimation`\
             is implemented, it cannot be calculated `total_frames`\
             which is necesary in the `progress_callback` function \


        Example taken from \
            https://matplotlib.org/stable/gallery/animation/dynamic_image.html

        Inputs
        ---------------------

        dirname: (str) directory where the video will be saved
        title_str: (str) title of the plot

        Returns
        --------------------

        `None`
        """
        ims = []

        num_images = T_transient.shape[1]

        fig, ax = plt.subplots()
        ax.set_title(title_str)
        ax.set_xlabel(self.xlabel_str)
        ax.set_ylabel(self.ylabel_str)

        for i_img in range(0, num_images):
            im = ax.scatter(points[:, 0], points[:, 1],
                            c=T_transient[:, i_img])
            if i_img == 0:
                ax.scatter(points[:, 0], points[:, 1],
                           c=T_transient[:, i_img])
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Temperature °C')

            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, blit=True)

        writer = animation.FFMpegWriter(fps=120, metadata=dict(artist='Me'),
                                        bitrate=1800)

        ani.save(dirname+filename, writer=writer,
                 progress_callback=progress_callback)

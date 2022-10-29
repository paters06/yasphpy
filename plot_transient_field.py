import pandas as pd
import time
from plotter import Plotter


def main():
    start = time.time()
    img_directory = '../yasphpy_images/'

    filename1 = 'particle_list.xlsx'
    filename2 = 'transient_state_results.xlsx'

    df1 = pd.read_excel(img_directory + filename1, sheet_name='Sheet1')
    particle_list = df1.to_numpy()

    df1 = pd.read_excel(img_directory + filename2, sheet_name='Sheet1')
    T_field = df1.to_numpy()

    end = time.time()

    print("Loading time: {:.2f} s".format(end-start))

    filename = 'heat_condution.mp4'

    field_plot = Plotter('x', 'y')
    field_plot.make_transient_plot(particle_list, T_field,
                                   img_directory,
                                   filename,
                                   'SPH temperature field. T_SPH')

    end2 = time.time()

    print("Creating video: {:.2f} s".format(end2-start))


if __name__ == '__main__':
    main()

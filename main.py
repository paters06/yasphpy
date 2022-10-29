import os
import numpy as np
import pandas as pd

from numerical_field import initial_conditions
from plotter import Plotter
from particles import Particles
from postprocessing import (analytical_solution, numerical_error)
from profiling_script import profiling_script
from sph_solver import sph_solver
import time


def main():
    start = time.time()

    print('Preprocessing')
    Lx = 1.0
    Ly = 1.0
    num_particles_side = 30
    field_particles = Particles(Lx, Ly, num_particles_side)
    field_particles.create_particles()
    field_particles.compute_masses()

    K = 1.0
    rho = 1.0
    cv = 1.0

    Ts = 100.0

    T_x0 = Ts

    T_0y = 0.0
    T_1y = 0.0

    T_x1 = 0.0
    T_xy = 0.0

    dt = 1.0e-3
    t_final = 1.0
    time_info = [dt, t_final]

    delta_stop = 1e-3

    particle_list = field_particles.get_particle_list()
    T_initial = initial_conditions(particle_list,
                                   T_0y, T_1y, T_x0, T_x1, T_xy)

    print('Solver')
    steady_state = False
    T_field = sph_solver(field_particles, time_info, T_initial, K, rho, cv,
                         delta_stop, steady_state)

    print('Postprocessing')
    T_exact = analytical_solution(particle_list, Ts)
    absolute_error = numerical_error(T_field, T_exact)
    print('Maximum temperature difference: {:.3f} °C'
          .format(np.max(absolute_error)))

    end = time.time()

    print("Execution time: {:.2f} s".format(end-start))

    if steady_state:
        field_plot = Plotter('x', 'y')
        field_plot.plot_scatter('Exact temperature field. T_e',
                                particle_list, T_exact)
        steady_fig = field_plot.plot_scatter('SPH temperature field. T_SPH',
                                             particle_list, T_field)
        field_plot.plot_scatter('Absolute numerical error',
                                particle_list, absolute_error)
        field_plot.show_plots(False)

        img_directory = '../yasphpy_images/'
        filename = 'Steady-state heat field.png'
        Plotter.save_plot(img_directory, steady_fig, filename)
    else:
        img_directory = '../yasphpy_images/'

        df1 = pd.DataFrame(particle_list)
        filename1 = 'particle_list.xlsx'
        df1.to_excel(img_directory + filename1, index=False, header=False)

        df2 = pd.DataFrame(T_field)
        filename2 = 'transient_state_results.xlsx'
        df2.to_excel(img_directory + filename2, index=False, header=False)

    end2 = time.time()

    print("Execution time: {:.2f} s".format(end2-start))


if __name__ == "__main__":
    main()
    # profiling_script(main)

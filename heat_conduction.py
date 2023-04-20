import numpy as np
import pandas as pd
from customized_logger import CustomizedLogger
from particles import Particles
from plotter import Plotter
from postprocessing import (analytical_solution, numerical_error)
from time_integration import time_integrator
from timer import Timer


class HeatConduction:
    def __init__(self, is_steady: bool) -> None:
        """
        Parameters
        ------------
        is_steady: (bool) variable to check if steady or transient \
            calculation must be done
        """
        self.steady_state = is_steady
        self.custom_logger = CustomizedLogger()

        self.custom_logger.info('Beginning the SPH simulation')
        self.timer = Timer()

    def discretize_domain(self, Lx: float, Ly: float, num_particles_side: int):
        """
        Parameters
        ------------
        Lx: (float) Length in x direction
        Ly: (float) Length in y direction
        num_particles_side: (float) Number of particles per side

        Members
        ------------
        particle_positions: (np.ndarray) Positions of the particles

        Returns
        ------------
        field_particles: (`Particles`) object of the class `Particles`
        """
        self._print_preprocessing_banner()
        field_particles = Particles(Lx, Ly, num_particles_side)
        field_particles.create_particles()
        field_particles.compute_masses()

        self.particle_positions = field_particles.get_particle_list()

        return field_particles

    def define_initial_conditions(self,
                                  T_0y, T_1y, T_x0, T_x1, T_xy):
        """
        Parameters
        ------------

        T_0y: (float) Prescribed temperature at x = 0
        T_1y: (float) Prescribed temperature at x = 1
        T_x0: (float) Prescribed temperature at y = 0
        T_x1: (float) Prescribed temperature at y = 1
        T_xy: (float) Prescribed temperature over the whole domain

        Returns
        ------------

        `None`
        """
        self.Ts = T_x0

        num_particles = self.particle_positions.shape[0]

        idx_0 = np.where(self.particle_positions[:, 0] == 0.0)[0]
        idx_1 = np.where(self.particle_positions[:, 0] == 1.0)[0]

        idy_0 = np.where(self.particle_positions[:, 1] == 0.0)[0]
        idy_1 = np.where(self.particle_positions[:, 1] == 1.0)[0]

        self.T_field = np.full((num_particles, 1), T_xy, dtype='float64')
        self.T_field[idy_0, 0] = T_x0  # Ts
        self.T_field[idx_0, 0] = T_0y
        self.T_field[idx_1, 0] = T_1y
        self.T_field[idy_1, 0] = T_x1

    def calculate_time_integration(self, field_particles, time_info,
                                   K, rho, cv,
                                   delta_stop):

        self._print_solver_banner()
        T_field_integrated = time_integrator(field_particles, time_info,
                                             self.T_field, K, rho, cv,
                                             delta_stop, self.steady_state,
                                             self.custom_logger)

        self.T_field = T_field_integrated

        self.timer.measure_time(self.custom_logger)

    def calculate_absolute_error(self, Ts):
        T_exact = analytical_solution(self.particle_positions, Ts)
        absolute_error = numerical_error(self.T_field, T_exact)

        print('Maximum temperature difference: {:.3f} °C'
              .format(np.max(absolute_error)))

        self.custom_logger.info('Maximum temperature difference: {:.3f} °C'
                                .format(np.max(absolute_error)))

    def export_steady_results(self, show,
                              exact_solution_available=False):
        self._print_postprocessing_banner()

        field_plot = Plotter('x', 'y')
        steady_fig = field_plot.plot_scatter('SPH temperature field. T_SPH',
                                             self.particle_positions,
                                             self.T_field)

        if exact_solution_available:
            T_exact = analytical_solution(self.particle_positions, self.Ts)
            absolute_error = numerical_error(self.T_field, T_exact)
            field_plot.plot_scatter('Exact temperature field. T_e',
                                    self.particle_positions, T_exact)
            field_plot.plot_scatter('Absolute numerical error',
                                    self.particle_positions, absolute_error)

        field_plot.show_plots(show)

        img_directory = '../yasphpy_images/'
        filename = 'Steady-state heat field.png'
        Plotter.save_plot(img_directory, steady_fig, filename)

        self.timer.measure_time(self.custom_logger)
        self.custom_logger.debug('Finishing the SPH simulation')

    def export_transient_results(self):
        img_directory = '../yasphpy_images/'

        df1 = pd.DataFrame(self.particle_positions)
        filename1 = 'particle_list.xlsx'
        df1.to_excel(img_directory + filename1, index=False, header=False)

        df2 = pd.DataFrame(self.T_field)
        filename2 = 'transient_state_results.xlsx'
        df2.to_excel(img_directory + filename2, index=False, header=False)

    def get_field_temperature(self):
        return self.T_field

    def set_temperature_field(self, T_field):
        self.T_field = T_field

    def _print_preprocessing_banner(self):
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('**           PREPROCESSING           **')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')

    def _print_solver_banner(self):
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('**              SOLVER               **')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')

    def _print_postprocessing_banner(self):
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('**          POSTPROCESSING           **')
        self.custom_logger.info('***************************************')
        self.custom_logger.info('***************************************')

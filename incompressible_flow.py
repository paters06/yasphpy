import math
import numpy as np
from customized_logger import CustomizedLogger
from open_periodic_boundary import OpenPeriodicBoundary
from reflective_boundary import ReflectiveBoundary
from particles import Particles
from plotter import Plotter
from timer import Timer

from fortran_src.neighbour import find_neighbour
from fortran_src.kernel_functions import smoothing_functions
from fortran_src.pressure_gradient import pressure_gradient
from fortran_src.mass_conservation import mass_conservation
from fortran_src.momentum_conservation import momentum_conservation
from fortran_src.virtual_repulsion_force import virtual_repulsion_force


class IncompressibleFlow:
    def __init__(self, L: float, D: float,
                 v_max: float, nu: float, rho: float) -> None:
        """
        L: length of the pipe
        D: pipe diameter
        v_max: maximum velocity of the fluid
        nu: dynamic viscosity of the fluid
        """
        num_particles_y = 50
        num_particles_x = int(L/D)*num_particles_y

        self.L = L
        self.D = D
        self.v_max = v_max
        self.nu = nu
        self.rho = rho
        self.c = 10.0
        self.n_dim = 2

        part1 = Particles(L, D, num_particles_x, num_particles_y, rho)

        self.particle_positions = part1.get_domain_particles()
        self.particle_densities = part1.get_particle_densities()
        self.particle_masses = part1.get_particle_masses()
        self.virtual_particles_I = part1.get_virtual_particles_I()
        self.virtual_particles_II = part1.get_virtual_particles_II()

        self.num_domain_particles = part1.get_num_domain_particles()
        self.num_virtual_particles_I = part1.get_num_virtual_particles_I()
        

        self.dx = part1.get_dx()
        self.dy = part1.get_dy()

        self.velocity_field = np.zeros((self.num_domain_particles, 2))
        self.pressure_field = (self.c**2)*self.particle_densities.copy()

        self.surface_tension_coefficient = 1.0

        solid_walls = ['left', 'right', 'down']

        self.custom_logger = CustomizedLogger()
        self.custom_logger.info('Beginning the SPH simulation')
        self.timer = Timer()

        self.virtual_velocity = self._enforce_velocity_BC()
        self.virtual_densities = part1.get_virtual_densities()
        self.virtual_masses = part1.get_virtual_masses()
        self.virtual_pressure = (self.c**2)*self.virtual_densities.copy()

        # print(self.pressure_field)
        # print(self.virtual_pressure)

        # self._enforce_pressure_BC()

        self._plot_particles()
        # self._plot_pressure_field()
        # self._plot_initial_BC()

    def _calculate_velocity_profile(self, y: float):
        # ux = 4*self.v_max*(y/self.D**2)*(self.D - y)
        ux = self.v_max
        return ux

    def _enforce_velocity_BC(self):
        virtual_particles_velocity = np.zeros_like(self.virtual_particles_I)
        for i in range(self.num_virtual_particles_I):
            y = self.virtual_particles_I[i, 1]

            if abs(y - self.D) < 1e-5:
                virtual_particles_velocity[i, 0] = self.v_max
                virtual_particles_velocity[i, 1] = 0.0

        return virtual_particles_velocity

    def _enforce_pressure_BC(self):
        """
        Modify this function when needed
        """
        P_inlet = 0.0
        for i in range(self.num_domain_particles):
            x = self.particle_positions[i, 0]
            if abs(x) < 1e-5:
                self.pressure_field[i, 0] = 0.0
            if abs(x - self.L) < 1e-5:
                self.pressure_field[i, 0] = 0.0

    def _calculate_equation_of_state(self):
        self.pressure_field = (self.c**2)*self.particle_densities.copy()
        # self._enforce_pressure_BC()

    def calculate_density(self, i, distances, indices,
                          num_neighbours, h, dx, dy):
        X_i = self.particle_positions[i, :]
        rho_i = 0.0
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            X_j = self.particle_positions[neigh_idx, :]
            mass_j = self.particle_masses[neigh_idx, 0]

            aux = 0.0
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)
                aux = mass_j*W

            rho_i += aux

        return rho_i

    def color_field(self, i, distances, indices, num_neighbours, h):
        X_i = self.particle_positions[i, :]
        color = 0.0
        color_grad = 0.0
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            X_j = self.particle_positions[neigh_idx, :]
            mass_j = self.particle_masses[neigh_idx]
            rho_j = self.particle_densities[neigh_idx]

            aux = 0.0
            aux_grad = 0.0
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)
                aux = (mass_j/rho_j)*W
                aux_grad = (mass_j/rho_j)*grad_W

            color += aux
            color_grad += aux_grad

        return color, color_grad

    def compute_surface_forces(self, i, distances, indices,
                               num_neighbours, h):
        X_i = self.particle_positions[i, :]
        cs_i, n_i = self.color_field(i, distances, indices, num_neighbours,
                                     h)
        n_i_norm = np.linalg.norm(n_i)
        color_laplacian = 0.0
        aux = 0.0
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            X_j = self.particle_positions[neigh_idx, :]
            mass_j = self.particle_masses[neigh_idx]
            rho_j = self.particle_densities[neigh_idx]
            cs_j, n_j = self.color_field(neigh_idx, distances, indices,
                                         num_neighbours, h)
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)
                aux = ((mass_j/rho_j)*(cs_i - cs_j)*(1.0/(r_ij**2)) *
                       np.dot((X_i - X_j), grad_W))
            else:
                aux = 0.0
            color_laplacian += aux

        fs_i = -self.surface_tension_coefficient*color_laplacian*(n_i/n_i_norm)
        return fs_i

    def solve_fluid_equations(self):
        # num_domain_particles = particles.get_num_domain_particles()
        # num_boundary_particles = particles.get_num_boundary_particles()

        m = self.num_domain_particles
        n = self.n_dim
        p = self.n_dim

        virtual_m = self.num_virtual_particles_I
        virtual_n = self.n_dim
        virtual_p = self.n_dim

        influence_radius = 2.5*self.dx
        k = 2.0
        h = influence_radius/k

        current_time = 0.0
        max_time = 1.0
        time_step = 0.001
        max_time_loop = max_time + time_step
        time_iterations = np.arange(current_time, max_time_loop, time_step)

        # boundary = OpenPeriodicBoundary(0.0, self.L, 0.0, self.D)
        reflective_walls = ['left', 'right', 'up', 'down']
        boundary = ReflectiveBoundary([self.L, self.D], reflective_walls,
                                      1.0, time_step)

        for time_i in time_iterations:
            self.custom_logger.debug('Time: {:.4f} s'.format(time_i))
            print('{:.3f} s/{:.3f} s'.format(time_i, max_time))
            for i_dom in range(0, self.num_domain_particles):
                # print('{:d}/{:d}'.format(i, self.num_domain_particles))
                # i_dom = i + self.num_boundary_particles
                X_i = self.particle_positions[i_dom, :]
                P_i = self.pressure_field[i_dom, 0]
                rho_i = self.particle_densities[i_dom, 0]
                vel_i = self.velocity_field[i_dom, :]

                domain_distances, domain_indices, num_domain_neighbours = \
                    find_neighbour(self.particle_positions,
                                   X_i, influence_radius, m, n, p)

                virtual_distances, virtual_indices, num_virtual_neighbours = \
                    find_neighbour(self.virtual_particles_I, X_i,
                                   influence_radius,
                                   virtual_m, virtual_n, virtual_p)

                drho_dt = mass_conservation(h, X_i, vel_i,
                                            num_domain_neighbours,
                                            domain_indices, domain_distances,
                                            num_virtual_neighbours,
                                            virtual_indices, virtual_distances,
                                            self.particle_masses,
                                            self.particle_positions,
                                            self.velocity_field,
                                            self.virtual_masses,
                                            self.virtual_particles_I,
                                            self.virtual_velocity,
                                            self.num_domain_particles,
                                            self.num_virtual_particles_I)

                dv_dt = momentum_conservation(h, X_i, P_i, rho_i, self.nu,
                                              num_domain_neighbours,
                                              domain_indices, domain_distances,
                                              num_virtual_neighbours,
                                              virtual_indices,
                                              virtual_distances,
                                              self.particle_masses,
                                              self.particle_positions,
                                              self.particle_densities,
                                              self.pressure_field,
                                              self.virtual_masses,
                                              self.virtual_particles_I,
                                              self.virtual_densities,
                                              self.virtual_pressure,
                                              self.num_domain_particles,
                                              self.num_virtual_particles_I)

                F_iv = virtual_repulsion_force(self.dx, X_i,
                                               self.v_max,
                                               num_virtual_neighbours,
                                               virtual_indices,
                                               self.virtual_particles_I,
                                               self.num_virtual_particles_I)

                # print(F_iv)
                dv_dt += F_iv

                self.particle_densities[i_dom, :] += (drho_dt*time_step)
                self.velocity_field[i_dom, :] += (dv_dt*time_step)

                # self.compute_collision_detection(i_dom, boundary)

            self._calculate_equation_of_state()

        # print(self.particle_densities)
        # self._plot_initial_BC()
        # self._plot_density_field()
        # self._plot_pressure_field()
        # self._plot_velocity_field()
        self._plot_velocity_magnitude()

    def compute_collision_detection(self, i_dom: int,
                                    boundary: ReflectiveBoundary):
        radius = 0.01

        Co = self.particle_positions[i_dom, :].copy()
        vo = self.velocity_field[i_dom, :].copy()
        Cf, vf = boundary.analyze_movement(Co, vo, radius)

        self.particle_positions[i_dom, :] = Cf.copy()
        self.velocity_field[i_dom, :] = vf.copy()

    def _plot_particles(self):
        field_plot = Plotter('x', 'y')
        initial_pos = field_plot.plot_points('Particles',
                                             self.particle_positions,
                                             self.virtual_particles_I,
                                             self.virtual_particles_II)
        field_plot.show_plots(True)

    def _plot_initial_BC(self):
        vec_field = self.velocity_field
        vvec_field = self.virtual_velocity
        mag_field = np.sqrt(vec_field[:, 0]**2 + vec_field[:, 1]**2)
        magv_field = np.sqrt(vvec_field[:, 0]**2 + vvec_field[:, 1]**2)
        field_plot = Plotter('x', 'y')
        vel_fig = field_plot.plot_scalar_field('Initial velocity field',
                                               'Velocity (m/s)',
                                               self.particle_positions,
                                               mag_field)
        # vvel_fig = field_plot.plot_scalar_field('Initial velocity field',
        #                                         'Velocity (m/s)',
        #                                         self.virtual_particles_I,
        #                                         magv_field)
        field_plot.show_plots(True)

    def _plot_density_field(self):
        field_plot = Plotter('x', 'y')
        steady_fig = field_plot.plot_scalar_field(r'Density field. $\rho$',
                                                  'Density (kg/m3)',
                                                  self.particle_positions,
                                                  self.particle_densities)

        field_plot.show_plots(True)

    def _plot_velocity_field(self):
        field_plot = Plotter('x', 'y')
        vel_fig = field_plot.plot_vector_field('Initial velocity field',
                                               'Velocity (m/s)',
                                               self.particle_positions,
                                               self.velocity_field)
        field_plot.show_plots(True)

    def _plot_streamline_field(self):
        field_plot = Plotter('x', 'y')
        vel_fig = field_plot.plot_stream_field('Initial velocity field',
                                               'Velocity (m/s)',
                                               self.particle_positions,
                                               self.velocity_field)
        field_plot.show_plots(True)

    def _plot_velocity_magnitude(self):
        vec_field = self.velocity_field
        mag_field = np.sqrt(vec_field[:, 0]**2 + vec_field[:, 1]**2)
        field_plot = Plotter('x', 'y')
        vel_fig = field_plot.plot_scalar_field('Initial velocity field',
                                               'Velocity (m/s)',
                                               self.particle_positions,
                                               mag_field)
        field_plot.show_plots(True)

    def _plot_pressure_field(self):
        field_plot = Plotter('x', 'y')
        steady_fig = field_plot.plot_scalar_field('Pressure field. P_field',
                                                  'Pressure (kPa)',
                                                  self.particle_positions,
                                                  self.pressure_field/1000)

        field_plot.show_plots(True)

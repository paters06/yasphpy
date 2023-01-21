import math
import numpy as np
from customized_logger import CustomizedLogger
from open_periodic_boundary import OpenPeriodicBoundary
from reflective_boundary import ReflectiveBoundary
from particles import Particles
from plotter import Plotter
from timer import Timer

import fortran_src.neighbour
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

        part1 = Particles(L, D, num_particles_x, num_particles_y, rho)

        # self.particle_positions = part1.get_particle_list()
        self.particle_positions = part1.get_domain_particle_list()
        self.particle_densities = part1.get_particle_densities()
        self.particle_masses = part1.get_particle_masses()
        self.virtual_particles = part1.get_virtual_particles_list()
        num_particles = part1.get_num_domain_particles()
        self.velocity_field = np.zeros((num_particles, 2))
        # self.pressure_field = np.zeros((num_particles, 1))
        self.pressure_field = (self.c**2)*self.particle_densities.copy()

        self.num_domain_particles = part1.get_num_domain_particles()
        self.num_boundary_particles = part1.get_num_boundary_particles()
        self.num_particles = part1.get_num_domain_particles()
        self.num_virtual_particles = part1.get_num_virtual_particles()

        self.dx = part1.get_dx()
        self.dy = part1.get_dy()

        self.surface_tension_coefficient = 1.0

        reflective_walls = ['left', 'right', 'up', 'down']

        self.custom_logger = CustomizedLogger()
        self.custom_logger.info('Beginning the SPH simulation')
        self.timer = Timer()

        self._enforce_velocity_BC()
        # self._enforce_pressure_BC()
        # self._plot_particles()
        # self._plot_velocity_field()
        self._plot_velocity_magnitude()
        # self._plot_pressure_field()

        # self.solve_fluid_equations()

    def _calculate_velocity_profile(self, y: float):
        # ux = 4*self.v_max*(y/self.D**2)*(self.D - y)
        ux = self.v_max
        return ux

    def _enforce_velocity_BC(self):
        num_particles = self.particle_positions.shape[0]
        for i in range(num_particles):
            x = self.particle_positions[i, 0]
            y = self.particle_positions[i, 1]
            # if abs(x - self.D) < 1e-5:
            #     ux = self._calculate_velocity_profile(y)
            #     self.velocity_field[i, 0] = ux
            #     self.velocity_field[i, 1] = 0.0

            # if abs(y) < 1e-5:
            #     self.velocity_field[i, 0] = 0.0
            #     self.velocity_field[i, 1] = 0.0

            if abs(y - self.D) < 1e-5:
                self.velocity_field[i, 0] = self.v_max
                self.velocity_field[i, 1] = 0.0

    def _enforce_pressure_BC(self):
        """
        Modify this function when needed
        """
        num_particles = self.particle_positions.shape[0]
        P_inlet = 0.0
        for i in range(num_particles):
            x = self.particle_positions[i, 0]
            if abs(x) < 1e-5:
                self.pressure_field[i, 0] = 0.0
            if abs(x - self.L) < 1e-5:
                self.pressure_field[i, 0] = 0.0

    def _calculate_equation_of_state(self):
        self.pressure_field = (self.c**2)*self.particle_densities.copy()
        # self._enforce_pressure_BC()

    def calculate_pressure_gradient(self, i, distances,
                                    indices, num_neighbours, h):
        X_i = self.particle_positions[i, :]
        P_i = self.pressure_field[i, 0]
        rho_i = self.particle_densities[i, 0]

        aux = np.zeros((2,))

        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            mass_j = self.particle_masses[neigh_idx, 0]

            aux1 = np.zeros((2,))

            if r_ij > 1e-5:
                X_j = self.particle_positions[neigh_idx, :]
                P_j = self.pressure_field[neigh_idx, 0]
                rho_j = self.particle_densities[neigh_idx, 0]

                W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)

                aux1 = mass_j * \
                    ((P_i/rho_i**2) + (P_j/rho_j**2))*grad_W

            aux[0] += aux1[0]
            aux[1] += aux1[1]

        pressure_gradient = -aux.copy()

        return pressure_gradient

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

    def solve_mass_conservation_eq(self, i, distances,
                                   indices, num_neighbours, h):
        drho_dt = 0.0
        X_i = self.particle_positions[i, :]
        vel_i = self.velocity_field[i, :]
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            X_j = self.particle_positions[neigh_idx, :]
            mass_j = self.particle_masses[neigh_idx, 0]
            vel_j = self.velocity_field[neigh_idx, :]

            aux = 0.0
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)
                aux = -mass_j*np.dot((vel_i - vel_j), grad_W)

            drho_dt += aux

        return drho_dt

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

    def solve_momentum_conservation_eq(self, i, distances,
                                       indices, num_neighbours,
                                       h, pressure_gradient):
        dv_dt = np.zeros((2,))
        X_i = self.particle_positions[i, :]

        viscous_forces = np.zeros((2,))

        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            mass_j = self.particle_masses[neigh_idx, 0]
            rho_j = self.particle_densities[neigh_idx, 0]
            X_j = self.particle_positions[neigh_idx, :]
            dist_ij = math.sqrt((X_i[0] - X_j[0])**2 + (X_i[1] - X_j[1])**2)

            aux = np.zeros((2,))

            W, dWdq, grad_W = smoothing_functions.cubic_kernel(h, X_i, X_j)
            if dist_ij > 1e-5:
                aux = 2*self.nu*(mass_j/rho_j)*((X_i - X_j)/dist_ij)*grad_W

            viscous_forces += aux

        # fs_i = self.compute_surface_forces(i, distances, indices,
        #                                    num_neighbours, h, dx, dy)

        dv_dt = viscous_forces
        # dv_dt = pressure_gradient + viscous_forces + fs_i

        return dv_dt

    def solve_fluid_equations(self):
        # num_domain_particles = particles.get_num_domain_particles()
        # num_boundary_particles = particles.get_num_boundary_particles()

        num_particles = self.num_particles
        m = self.particle_positions.shape[0]
        n = self.particle_positions.shape[1]
        p = n

        virtual_m = self.num_virtual_particles
        virtual_n = self.virtual_particles.shape[1]
        virtual_p = virtual_n

        influence_radius = 2.5*self.dx
        k = 2.0
        h = influence_radius/k

        current_time = 0.0
        max_time = 1.0
        time_step = 0.01
        max_time_loop = max_time + time_step
        time_iterations = np.arange(current_time, max_time_loop, time_step)

        # boundary = OpenPeriodicBoundary(0.0, self.L, 0.0, self.D)
        reflective_walls = ['left', 'right', 'up', 'down']
        boundary = ReflectiveBoundary([self.L, self.D], reflective_walls,
                                      1.0, time_step)

        for time_i in time_iterations:
            self.custom_logger.debug('Time: {:.4f} s'.format(time_i))
            print('{:.3f} s/{:.3f} s'.format(time_i, max_time))
            for i in range(0, self.num_domain_particles):
                # for i in range(0, self.num_particles):
                # print('{:d}/{:d}'.format(i, self.num_domain_particles))
                # i_dom = i + self.num_boundary_particles
                i_dom = i
                X_i = self.particle_positions[i_dom, :]
                P_i = self.pressure_field[i_dom, 0]
                rho_i = self.particle_densities[i_dom, 0]
                vel_i = self.velocity_field[i_dom, :]

                distances, indices, num_neighbours = \
                    neighbour.find_neighbour(self.particle_positions,
                                             X_i,
                                             influence_radius,
                                             m, n, p)

                virtual_distances, virtual_indices, num_virtual_neighbours = \
                    neighbour.find_neighbour(self.virtual_particles,
                                             X_i,
                                             influence_radius,
                                             virtual_m, virtual_n, virtual_p)

                # print(self.particle_positions.shape)
                # print(num_particles)
                grad_P = pressure_gradient(h, X_i, P_i, rho_i, num_neighbours,
                                           indices, distances,
                                           self.particle_positions,
                                           self.particle_masses,
                                           self.particle_densities,
                                           self.pressure_field,
                                           num_particles)

                drho_dt = mass_conservation(h, X_i, vel_i, num_neighbours,
                                            indices, distances,
                                            self.particle_masses,
                                            self.particle_positions,
                                            self.velocity_field, num_particles)

                dv_dt = momentum_conservation(h, X_i, self.nu, num_neighbours,
                                              indices, distances,
                                              self.particle_masses,
                                              self.particle_densities,
                                              self.particle_positions,
                                              num_particles)

                F_iv = virtual_repulsion_force(influence_radius, X_i,
                                               self.v_max,
                                               num_virtual_neighbours,
                                               virtual_indices,
                                               self.virtual_particles,
                                               self.num_virtual_particles)

                # grad_P = self.calculate_pressure_gradient(i,
                #                                           distances,
                #                                           indices,
                #                                           num_neighbours,
                #                                           h)

                # drho_dt = self.solve_mass_conservation_eq(i,
                #                                           distances,
                #                                           indices,
                #                                           num_neighbours,
                #                                           h)

                # dv_dt = self.solve_momentum_conservation_eq(i,
                #                                             distances,
                #                                             indices,
                #                                             num_neighbours,
                #                                             h, grad_P)

                # print(drho_dt.shape)
                self.particle_densities[i_dom, :] += (drho_dt*time_step)
                self.velocity_field[i_dom, :] += (dv_dt*time_step)

                self.compute_collision_detection(i_dom, boundary)

                # self.particle_positions[i_dom, :] += (self.velocity_field[i_dom, :]*time_step)
                # updated_pos = self.particle_positions[i_dom, :]
                # updated_vel = self.velocity_field[i_dom, :]
                # new_pos, new_vel = boundary.check_particle_in_domain(updated_pos, updated_vel)
                # self.particle_positions[i_dom, :] = new_pos.copy()
                # self.velocity_field[i_dom, :] = new_vel.copy()
            # self._enforce_velocity_BC()
            # print(np.max(self.particle_positions[:, 1]))
            self._calculate_equation_of_state()
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
                                             self.virtual_particles)
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

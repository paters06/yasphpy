import math
import numpy as np
# from discrete_laplacian import discrete_laplacian
from kernel_functions import smoothing_functions
import neighbour
from customized_logger import CustomizedLogger
from particles import Particles
from plotter import Plotter
from reflective_boundary import ReflectiveBoundary
from timer import Timer


class FluidHydrostatics:

    def __init__(self, particles: Particles, nu: float, H: float) -> None:
        self.particle_positions = particles.get_particle_list()
        self.particle_densities = particles.get_particle_densities()
        self.particle_masses = particles.get_particle_masses()
        self.particle_velocities = particles.get_particle_velocities()
        self.nu = nu
        GRAVITY = 9.81
        self.gravity_acc = GRAVITY

        self.atmospheric_pressure = 101325.0
        self.fluid_density = 1000.0
        self.pressure_field = 0.
        self.surface_tension_coefficient = 1.0

        self.free_surface_height = H

        self.custom_logger = CustomizedLogger()
        self.custom_logger.info('Beginning the SPH simulation')
        self.timer = Timer()

        num_particles = particles.get_num_particles()
        self.P_mod = np.zeros((num_particles, 1))

    def calculate_modified_pressure(self):
        H = self.free_surface_height
        self.P_mod = (self.pressure_field - self.atmospheric_pressure) - \
            (self.fluid_density*self.gravity_acc *
                (H - self.particle_positions[:, 1]))

    def calculate_absolute_pressure(self):
        Po = self.atmospheric_pressure
        rho = self.fluid_density
        g = self.gravity_acc
        H = self.free_surface_height
        y = self.particle_positions[:, 1]

        self.P_abs = Po + self.P_mod + rho*g*(H - y)

    def calculate_pressure_gradient(self, i, distances,
                                    indices, num_neighbours,
                                    h, dx, dy):
        # P_mod_i = self.P_mod[i]
        P_mod_i = 0.0
        rho_i = self.particle_densities[i, 0]

        aux = np.zeros((2,))

        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            mass_j = self.particle_masses[neigh_idx, 0]

            aux1 = np.zeros((2,))

            if r_ij > 1e-5:
                # P_mod_j = self.P_mod[neigh_idx]
                P_mod_j = 0.0
                rho_j = self.particle_densities[neigh_idx, 0]

                W, dWdq, grad_W = smoothing_functions.cubic_kernel(r_ij, h,
                                                                   dx, dy)

                # temp = mass_j*((P_mod_i/rho_i**2) + (P_mod_j/rho_j**2))
                # aux2 = temp*grad_W

                aux1 = mass_j * \
                    ((P_mod_i/rho_i**2) + (P_mod_j/rho_j**2)) * \
                    grad_W

            aux[0] += aux1[0]
            aux[1] += aux1[1]

        pressure_gradient = -rho_i*aux

        return pressure_gradient

    def solve_mass_conservation_eq(self, i, distances,
                                   indices, num_neighbours,
                                   h, dx, dy):
        drho_dt = 0.
        # print(drho_dt)
        vel_i = self.particle_velocities[i, :]
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            mass_j = self.particle_masses[neigh_idx]
            vel_j = self.particle_velocities[neigh_idx, :]

            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(r_ij, h,
                                                                   dx, dy)
                # print(grad_W)
                aux = -mass_j*np.dot((vel_i - vel_j), grad_W)
            else:
                aux = 0.0

            drho_dt += aux

        # print(drho_dt)
        return drho_dt

    def color_field(self, i, distances, indices, num_neighbours, h, dx, dy):
        X_i = self.particle_positions[i, :]
        color = 0.0
        color_grad = 0.0
        for j in range(0, num_neighbours):
            neigh_idx = indices[j] - 1
            r_ij = distances[j]
            X_j = self.particle_positions[neigh_idx, :]
            mass_j = self.particle_masses[neigh_idx]
            rho_j = self.particle_densities[neigh_idx]
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(r_ij, h,
                                                                   dx, dy)
                aux = (mass_j/rho_j)*W
                aux_grad = (mass_j/rho_j)*grad_W
            else:
                aux = 0.0
                aux_grad = 0.0

            color += aux
            color_grad += aux_grad

        return color, color_grad

    def compute_surface_forces(self, i, distances, indices, num_neighbours,
                               h, dx, dy):
        X_i = self.particle_positions[i, :]
        cs_i, n_i = self.color_field(i, distances, indices, num_neighbours,
                                     h, dx, dy)
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
                                         num_neighbours, h, dx, dy)
            if r_ij > 1e-5:
                W, dWdq, grad_W = smoothing_functions.cubic_kernel(r_ij, h,
                                                                   dx, dy)
                aux = ((mass_j/rho_j)*(cs_i - cs_j)*(1.0/(r_ij**2)) *
                       np.dot((X_i - X_j), grad_W))
            else:
                aux = 0.0
            color_laplacian += aux

        fs_i = -self.surface_tension_coefficient*color_laplacian*(n_i/n_i_norm)
        return fs_i

    def solve_momentum_conservation_eq(self, i, distances,
                                       indices, num_neighbours,
                                       h, dx, dy,
                                       pressure_gradient):
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

            W, dWdq, grad_W = smoothing_functions.cubic_kernel(r_ij, h, dx, dy)
            if dist_ij > 1e-5:
                aux = 2*self.nu*(mass_j/rho_j)*((X_i - X_j)/dist_ij)\
                    * grad_W

            viscous_forces += aux

        fs_i = self.compute_surface_forces(i, distances, indices,
                                           num_neighbours, h, dx, dy)

        dv_dt = pressure_gradient + viscous_forces + fs_i

        return dv_dt

    def update_densities_velocities(self, particles: Particles):
        num_domain_particles = particles.get_num_domain_particles()
        num_boundary_particles = particles.get_num_boundary_particles()

        m = self.particle_positions.shape[0]
        n = self.particle_positions.shape[1]
        p = n

        dx = particles.get_dx()
        dy = particles.get_dy()
        influence_radius = 2.5*dx
        k = 2.0
        h = influence_radius/k

        current_time = 0.
        max_time = 5.0
        time_step = 0.01
        max_time_loop = max_time + time_step
        time_iterations = np.arange(current_time, max_time_loop, time_step)

        bounding_box = [particles.Lx, particles.Ly]
        CR = 1.0
        boundary = ReflectiveBoundary(bounding_box, CR, time_step)

        for time_i in time_iterations:
            self.custom_logger.debug('Time: {:.4f} s'.format(time_i))
            print(time_i)
            for i in range(0, num_domain_particles):
                i_dom = i + num_boundary_particles
                X_i = self.particle_positions[i_dom, :]

                distances, indices, num_neighbours = \
                    neighbour.find_neighbour(self.particle_positions,
                                             X_i,
                                             influence_radius,
                                             m,
                                             n,
                                             p)

                grad_P_mod = self.calculate_pressure_gradient(i,
                                                              distances,
                                                              indices,
                                                              num_neighbours,
                                                              h, dx, dy)

                drho_dt = self.solve_mass_conservation_eq(i,
                                                          distances,
                                                          indices,
                                                          num_neighbours,
                                                          h, dx, dy)

                dv_dt = self.solve_momentum_conservation_eq(i,
                                                            distances,
                                                            indices,
                                                            num_neighbours,
                                                            h, dx, dy,
                                                            grad_P_mod)

                # print(drho_dt.shape)
                # self.P_mod[i_dom, :] += (grad_P_mod*time_step)
                self.particle_densities[i_dom, :] += (drho_dt*time_step)
                self.particle_velocities[i_dom, :] += (dv_dt*time_step)

                self.compute_collision_detection(i_dom, boundary)

        # self.calculate_absolute_pressure()
        self._plot_modified_pressure()

    def compute_collision_detection(self, i_dom: int,
                                    boundary: ReflectiveBoundary):
        radius = 0.01

        Co = self.particle_positions[i_dom, :].copy()
        vo = self.particle_velocities[i_dom, :].copy()
        Cf, vf = boundary.analyze_movement(Co, vo, radius)

        self.particle_positions[i_dom, :] = Cf.copy()
        self.particle_velocities[i_dom, :] = vf.copy()

    def _plot_modified_pressure(self):
        field_plot = Plotter('x', 'y')
        steady_fig = field_plot.plot_scatter('Modified pressure. P_mod',
                                             self.particle_positions,
                                             self.P_mod)

        field_plot.show_plots(True)

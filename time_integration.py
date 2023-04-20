import logging
import numpy as np
import neighbour
from customized_logger import CustomizedLogger
from discrete_laplacian import discrete_laplacian


def time_integrator(field_particles, time_info: list, T_initial: np.ndarray,
                    K: float, rho: float, cv: float, delta_stop: float,
                    steady_result, custom_logger):
    """
    Parameters
    ---------------

        field_particles: (Particle class) Object that stores the \
             information about the particles
        time_info: (list) Information about the time steps \
             used in the simulation
        T_initial: (np.ndarray) Array with the numerical solution filled \
             with the initial conditions
        K: (float) Thermal conductivity
        rho: (float) Density
        cv: (float) Specific heat
        delta_stop: (float) tolerance for stopping the simulation
        steady_result: (bool) If `True`, the function returns only the \
            steady state result. Otherwise, it returns the transient matrix
        custom_logger: (CustomizedLogger) A custom logging object used to\
            print information to the console and to a given file

    Returns
    ---------------
        T_field: (np.ndarray) Numerical solution at the end of the \
            temporal integration
    """

    custom_logger.debug('Entering the time integration function')

    particle_list = field_particles.get_particle_list()
    particle_densities = field_particles.get_particle_densities()
    particle_masses = field_particles.get_particle_masses()

    dx = field_particles.get_dx()
    dy = field_particles.get_dy()

    num_particles = field_particles.get_num_particles()
    num_domain_particles = field_particles.get_num_domain_particles()
    num_boundary_particles = field_particles.get_num_boundary_particles()

    m = particle_list.shape[0]
    n = particle_list.shape[1]
    p = n

    dt = time_info[0]
    t_final = time_info[1]
    time_steps = np.arange(0, t_final, step=dt)
    max_steps = 5000

    T_field = T_initial.copy()
    T_transient = np.zeros((T_field.shape[0], 5000))

    laplacian = np.zeros((num_particles, 1), dtype='float')

    delta = 1e5
    tol = 1e-4
    i_time = 0
    current_time = 0.0
    # for i_time in range(0, num_steps):
    while delta > delta_stop:
        alpha_i = K/(rho*cv)

        influence_radius = 2.5*dx
        k = 2.0
        h = influence_radius/k

        for i in range(0, num_domain_particles):
            i_dom = i + num_boundary_particles
            X_i = particle_list[i_dom, :]
            T_i = T_field[i_dom, 0]

            distances, indices, num_neighbours = \
                neighbour.find_neighbour(particle_list,
                                         X_i,
                                         influence_radius,
                                         m,
                                         n,
                                         p)

            laplacian[i_dom, 0] = discrete_laplacian(alpha_i, h,
                                                     particle_masses,
                                                     particle_densities,
                                                     indices, distances,
                                                     num_neighbours,
                                                     T_field, T_i,
                                                     num_particles)

        max_delta_field = np.max(laplacian*dt)

        custom_logger.debug('Time: {:.4f} s'.format(current_time))
        custom_logger.debug('Maximum delta field: {:.5f}'
                            .format(max_delta_field))

        T_field += laplacian*dt
        T_transient[:, None, i_time] = T_field
        delta = max_delta_field
        current_time += dt
        i_time += 1

    custom_logger.debug('Leaving the time integration function')

    if steady_result:
        return T_field
    else:
        return T_transient[:, 0:i_time]

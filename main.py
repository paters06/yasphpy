import math
import numpy as np
import kernel as kn
# from matplotlib import pyplot as plt
from plotter import Plotter
from particles import Particles
import time


def domain_particle_creation(lx, ly, npx, npy):
    x = np.linspace(0, lx, npx)
    y = np.linspace(0, ly, npy)

    x_dom = x[1:-1]
    y_dom = y[1:-1]

    np_2d = (npx - 2)*(npy - 2)

    x_field, y_field = np.meshgrid(x_dom, y_dom)
    x_field_1d = np.reshape(x_field, (1, np_2d))
    y_field_1d = np.reshape(y_field, (1, np_2d))

    particles = np.vstack((x_field_1d, y_field_1d))

    return particles


def boundary_particle_creation(lx, ly, npx, npy):
    x = np.linspace(0, lx, 2*npx)
    y = np.linspace(0, ly, 2*npy)

    np_2d = 4*(npx)*(npy)

    x_field, y_field = np.meshgrid(x, y)
    x_field_1d = np.reshape(x_field, (1, np_2d))
    y_field_1d = np.reshape(y_field, (1, np_2d))

    idx_0 = np.where(x_field_1d[0, :, None] == 0.0)[0]
    idx_1 = np.where(x_field_1d[0, :, None] == 1.0)[0]

    idy_0 = np.where(y_field_1d[0, :, None] == 0.0)[0]
    idy_1 = np.where(y_field_1d[0, :, None] == 1.0)[0]

    id_bounds = np.stack((idx_0, idx_1, idy_0, idy_1))
    id_boundary = np.unique(id_bounds)

    x_boundary1d = x_field_1d[0, id_boundary]
    y_boundary1d = y_field_1d[0, id_boundary]

    boundary_particles = np.vstack((x_boundary1d, y_boundary1d))
    return boundary_particles


def neighbor_list(X_i, particle_list, influence_radius):
    neigh_idx = []
    num_particle = particle_list.shape[1]
    for j in range(0, num_particle):
        X_j = particle_list[:, j]
        d = np.linalg.norm(X_i - X_j)
        if abs(d) > 1e-5:
            if d < influence_radius:
                neigh_idx.append(j)

    return neigh_idx


def particle_density(i, h, particle_list, m_list, influence_radius):
    rho_i = 0.0
    X_i = particle_list[:, i]
    # num_particles = particle_list.shape[1]
    neighbor_indices = neighbor_list(X_i, particle_list, influence_radius)
    num_neighbors = neighbor_indices.shape[0]

    for j in range(0, num_neighbors):
        neigh_idx = neighbor_indices[j]
        X_j = particle_list[:, neigh_idx]
        rho_i += m_list[:, j]*kn.cubic_spline_kernel(X_i, X_j, h)

    return rho_i


def discretized_laplacian(K, rho, cv, dx, dy, m_list, rho_list,
                          particle_list, T_field, X_i, T_i):

    alpha_i = K/(rho*cv)

    influence_radius = 2.5*dx
    k = 2.0
    h = influence_radius/k

    neighbor_indices = neighbor_list(X_i, particle_list, influence_radius)
    num_neighbors = len(neighbor_indices)

    particle = 0.0

    for j in range(1, num_neighbors):
        neigh_idx = neighbor_indices[j]
        X_j = particle_list[:, neigh_idx]
        r_ij = np.linalg.norm(X_i - X_j)
        particle += ((m_list[:, neigh_idx]/rho_list[:, neigh_idx])
                     * (T_i - T_field[:, neigh_idx])
                     * kn.cubic_spline_kernel_derivative(X_i, X_j, h))

    laplace_i = 2.0*alpha_i*particle

    return laplace_i


def initial_conditions(particle_list, T_0y, T_1y, T_x0, T_x1, T_xy):

    num_particles = particle_list.shape[1]

    idx_0 = np.where(particle_list[0, :, None] == 0.0)[0]
    idx_1 = np.where(particle_list[0, :, None] == 1.0)[0]

    idy_0 = np.where(particle_list[1, :, None] == 0.0)[0]
    idy_1 = np.where(particle_list[1, :, None] == 1.0)[0]

    T_field = np.full((1, num_particles), T_xy, dtype='float64')
    T_field[:, idy_0] = T_x0  # Ts
    T_field[:, idx_0] = T_0y
    T_field[:, idx_1] = T_1y
    T_field[:, idy_1] = T_x1

    return T_field


def sph_solver(field_particles, time_info, T_initial, K, rho, cv):

    particle_list = field_particles.get_particle_list()
    particle_densities = field_particles.get_particle_densities()
    particle_masses = field_particles.get_particle_masses()

    dx = field_particles.get_dx()
    dy = field_particles.get_dy()

    num_particles = field_particles.get_num_particles()
    num_domain_particles = field_particles.get_num_domain_particles()
    num_boundary_particles = field_particles.get_num_boundary_particles()

    dt = time_info[0]
    t_final = time_info[1]
    time_steps = np.arange(0, t_final, step=dt)
    num_steps = len(time_steps)

    T_field = T_initial.copy()

    laplacian = np.zeros((1, num_particles), dtype='float')

    delta = 1e5
    tol = 1e-4
    current_time = 0.0
    for i_time in range(0, num_steps):
    # while delta > tol:
        print("Time: {:.4f} s".format(current_time))
        for i in range(0, num_domain_particles):
            # print("Particle #{:d}".format(i+1))
            i_dom = i + num_boundary_particles
            # print("Particle #{:d}".format(i_dom+1))
            X_i = particle_list[:, i_dom]
            T_i = T_field[:, i_dom]
            laplacian[0, i_dom] = discretized_laplacian(K, rho, cv, dx, dy,
                                                        particle_masses,
                                                        particle_densities,
                                                        particle_list, T_field,
                                                        X_i, T_i)
        max_delta_field = np.max(laplacian*dt)
        print("Maximum delta field: {:.5f}".format(max_delta_field))
        T_field += laplacian*dt
        delta = max_delta_field
        current_time += dt

    return T_field


def analytical_solution(particle_list, Ts):
    num_series_iter = 91
    num_particles = particle_list.shape[1]
    N = num_particles
    T_exact = np.zeros((1, num_particles))
    delta_T = np.zeros((1, num_particles))

    for i in range(0, num_particles):
        for N in range(1, num_series_iter):
            AN = ((2*Ts)/(N*math.pi))*(((-1)**N - 1)/(np.sinh(N*math.pi)))
            delta_T[0, i] = AN*np.sin(N*math.pi*particle_list[0, i])\
                * np.sinh(N*math.pi*(particle_list[1, i] - 1))
            T_exact += delta_T

    return T_exact


def numerical_error(T_numerical, T_analytical):
    return np.abs(T_analytical - T_numerical)


def postprocessing_fields(points, field):
    points_x = np.unique(points[0, :])
    points_y = np.unique(points[1, :])
    num_x = len(points_x)
    num_y = len(points_y)
    field_xy = np.reshape(field, (num_x, num_y))
    return points_x, points_y, field_xy


def main():
    start = time.time()

    print('Preprocessing')
    Lx = 1.0
    Ly = 1.0
    num_particles_side = 20
    field_particles = Particles(Lx, Ly, num_particles_side)
    field_particles.create_particles()
    field_particles.compute_masses()

    # num_particles_y = 25

    K = 1.0
    rho = 1.0
    cv = 1.0

    Ts = 100.0

    T_x0 = Ts

    T_0y = 0.0
    T_1y = 0.0

    T_x1 = 0.0
    T_xy = 0.0

    dt = 1.0e-2
    t_final = 1.0
    time_info = [dt, t_final]

    particle_list = field_particles.get_particle_list()
    T_initial = initial_conditions(particle_list,
                                   T_0y, T_1y, T_x0, T_x1, T_xy)

    print('Solver')
    T_field = sph_solver(field_particles, time_info, T_initial, K, rho, cv)

    print('Postprocessing')
    T_exact = analytical_solution(particle_list, Ts)
    absolute_error = numerical_error(T_field, T_exact)
    print('Maximum temperature difference: {:.3f} °C'
          .format(np.max(absolute_error)))

    end = time.time()

    print("Execution time: {:.2f} s".format(end-start))

    field_plot = Plotter('x', 'y')
    field_plot.plot_scatter('Exact temperature field. T_e', particle_list, T_exact)
    field_plot.plot_scatter('SPH temperature field. T_SPH', particle_list, T_field)
    field_plot.plot_scatter('Absolute numerical error', particle_list, absolute_error)
    field_plot.show_plots(True)


if __name__ == "__main__":
    main()

import math
import numpy as np
import kernel as kn
from matplotlib import pyplot as plt


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


def neighbor_list(i, particle_list, influence_radius):
    neigh_idx = []
    num_particle = particle_list.shape[1]
    X_i = particle_list[:, i]
    for j in range(0, num_particle):
        X_j = particle_list[:, j]
        d = np.linalg.norm(X_i - X_j)
        if d < influence_radius:
            neigh_idx.append(j)

    return particle_list[:, neigh_idx]


def particle_density(i, h, particle_list, m_list, influence_radius):
    rho_i = 0
    X_i = particle_list[:, i]
    num_particles = particle_list.shape[1]
    neighbor_particle = neighbor_list(i, particle_list, influence_radius)
    num_neighbors = neighbor_particle.shape[0]

    for j in range(0, num_neighbors):
        X_j = neighbor_particle[:, j]
        # print(kn.cubic_spline_kernel(X_i, X_j, h))
        rho_i += m_list[:, j]*kn.cubic_spline_kernel(X_i, X_j, h)

    # print(rho_i)
    return rho_i


def discretized_laplacian(K, rho, cv, dx, dy, m_list, rho_list,
                          particle_list, T_field, X_i, T_i):

    alpha_i = K/(rho*cv)

    influence_radius = 2.5*dx
    k = 2.0
    h = influence_radius/k

    num_particles = m_list.shape[1]
    # X_i = particle_list[:, i]

    particle = 0

    for j in range(1, num_particles):
        # rho_j = particle_density(j, h, particle_list,
        #                          m_list, influence_radius)
        X_j = particle_list[:, j]
        particle += ((m_list[:, j]/rho_list[:, j])
                     * (T_i - T_field[:, j])
                     * kn.cubic_spline_kernel_derivative(X_i, X_j, h))
        # print((m_list[:,j]/rho_list[:,j])*(T_field[:,i] - T_field[:,j])
        #                * kn.cubic_spline_kernel_derivative(X_i, X_j, h))

    # print(particle)
    laplace_i = 2*alpha_i*particle

    return laplace_i


def initial_conditions(particle_list, T_0y, T_1y, T_x0, T_x1, T_xy):
    # x = np.linspace(0, lx, npx)
    # y = np.linspace(0, ly, npy)

    # np_2d = npx*npy

    # x_field, y_field = np.meshgrid(x, y)
    # x_field_1d = np.reshape(x_field, (1, np_2d))
    # y_field_1d = np.reshape(y_field, (1, np_2d))

    num_particles = particle_list.shape[1]

    idx_0 = np.where(particle_list[0, :, None] == 0.0)[0]
    idx_1 = np.where(particle_list[0, :, None] == 1.0)[0]

    idy_0 = np.where(particle_list[1, :, None] == 0.0)[0]
    idy_1 = np.where(particle_list[1, :, None] == 1.0)[0]

    T_field = np.full((1, num_particles), T_xy)
    T_field[:, idy_0] = T_x0  # Ts
    T_field[:, idx_0] = T_0y
    T_field[:, idx_1] = T_1y
    T_field[:, idy_1] = T_x1

    return T_field


def analytical_solution(particle_list, Ts):
    num_series_iter = 21
    num_particles = particle_list.shape[1]
    N = num_particles
    T_exact = np.zeros((1, num_particles))
    delta_T = np.zeros((1, num_particles))

    for i in range(0, num_particles):
        for N in range(1, num_series_iter):
            AN = ((2*Ts)/(N*math.pi))*(((-1)**N - 1)/(np.sinh(N*math.pi)))
            delta_T[0, i] = AN*np.sin(N*math.pi*particle_list[0, i])\
                * np.sinh(N*math.pi*(particle_list[1, i] - 1))
            np.add(T_exact, delta_T, out=T_exact, casting="unsafe")

    return T_exact


def numerical_error(T_numerical, T_analytical):
    return np.abs(T_analytical - T_numerical)


def plot_field(points, field):
    plt.figure()
    plt.scatter(points[0, :], points[1, :], c=field)
    plt.colorbar()
    # plt.show()


def main():
    print('Preprocessing')
    num_particles_x = 20
    num_particles_y = 20
    # num_particles = num_particles_x*num_particles_y

    Lx = 1.0
    Ly = 1.0

    dx = Lx/num_particles_x
    dy = Ly/num_particles_y

    K = 1
    rho = 1
    cv = 1

    Ts = 100

    T_x0 = Ts

    T_0y = 0
    T_1y = 0

    T_x1 = 0
    T_xy = 0

    dt = 5e-2
    t_final = 1
    time_steps = np.arange(0, t_final, step=dt)
    num_steps = len(time_steps)

    domain_particle_list = domain_particle_creation(Lx, Ly, num_particles_x, num_particles_y)
    boundary_particle_list = boundary_particle_creation(Lx, Ly, num_particles_x, num_particles_y)
    particle_list = np.hstack((boundary_particle_list, domain_particle_list))

    num_particles = particle_list.shape[1]
    num_domain_particles = domain_particle_list.shape[1]
    num_boundary_particles = boundary_particle_list.shape[1]
    particle_densities = np.ones((1, num_particles))
    particle_masses = dx*dy*particle_densities
    T_initial = initial_conditions(particle_list,
                                   T_0y, T_1y, T_x0, T_x1, T_xy)

    T_field = T_initial.copy()

    print('Solver')
    laplacian = np.zeros((1, num_particles))

    for i_time in range(0, num_steps):
        print("Time iteration #{:d}".format(i_time+1))
        for i in range(0, num_domain_particles):
            # print("Particle #{:d}".format(i+1))
            i_dom = i + num_boundary_particles
            X_i = particle_list[:, i_dom]
            T_i = T_field[:, i_dom]
            laplacian[0, i_dom] = discretized_laplacian(K, rho, cv, dx, dy,
                                                        particle_masses,
                                                        particle_densities,
                                                        particle_list, T_field,
                                                        X_i, T_i)
        np.add(T_field, laplacian*dt, out=T_field, casting="unsafe")
        # print(laplacian.T)
        # plt.figure()
        # plt.scatter(particle_list[0, :], particle_list[1, :], c=T_field)
        # plt.colorbar()

    print('Postprocessing')
    print(T_field)

    T_exact = analytical_solution(particle_list, Ts)
    # print(T_exact)

    interpolation_error = numerical_error(T_field, T_exact)
    print('Maximum temperature difference: {:.3f} °C'
          .format(np.max(interpolation_error)))

    plot_field(particle_list, T_exact)
    plot_field(particle_list, T_field)
    plot_field(particle_list, interpolation_error)

    print("Remember using one `plt.show' line")

    # plt.figure()
    # plt.scatter(particle_list[0, :], particle_list[1, :], c=T_exact)
    # plt.colorbar()
    # plt.figure()
    # plt.scatter(particle_list[0, :], particle_list[1, :], c=T_field)
    # plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()

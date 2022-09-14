import math
import numpy as np
import kernel as kn
from matplotlib import pyplot as plt


def particle_creation(lx, ly, npx, npy):
    x = np.linspace(0, lx, npx)
    y = np.linspace(0, ly, npy)

    np_2d = npx*npy

    x_field, y_field = np.meshgrid(x, y)
    x_field_1d = np.reshape(x_field, (1, np_2d))
    y_field_1d = np.reshape(y_field, (1, np_2d))

    particles = np.vstack((x_field_1d, y_field_1d))

    return particles


def neighbor_list(i, particle_list, influence_radius):
    neigh_idx = []
    num_particle = particle_list.shape[1]
    X_i = particle_list[:,i]
    for j in range(0, num_particle):
        X_j = particle_list[:,j]
        d = np.linalg.norm(X_i - X_j)
        if d < influence_radius:
            neigh_idx.append(j)

    return particle_list[:,neigh_idx]


def particle_density(i, h, particle_list, m_list, influence_radius):
    rho_i = 0
    X_i = particle_list[:,i]
    num_particles = particle_list.shape[1]
    neighbor_particle = neighbor_list(i, particle_list, influence_radius)
    num_neighbors = neighbor_particle.shape[0]

    for j in range(0, num_neighbors):
        X_j = neighbor_particle[:,j]
        # print(kn.cubic_spline_kernel(X_i, X_j, h))
        rho_i += m_list[:,j]*kn.cubic_spline_kernel(X_i, X_j, h)

    # print(rho_i)
    return rho_i


def discretized_laplacian(K, rho, cv, dx, dy, m_list, rho_list,
                          particle_list, T_field, i):
    
    alpha_i = K/(rho*cv)

    influence_radius = 2.5*dx
    k = 2.0
    h = influence_radius/k

    num_particles = m_list.shape[1]
    X_i = particle_list[:,i]

    particle = 0

    for j in range(1, num_particles):
        # rho_j = particle_density(j, h, particle_list, m_list, influence_radius)
        X_j = particle_list[:,j]
        particle += ((m_list[:,j]/rho_list[:,j])*(T_field[:,i] - T_field[:,j])
                      * kn.cubic_spline_kernel_derivative(X_i, X_j, h))
        # print((m_list[:,j]/rho_list[:,j])*(T_field[:,i] - T_field[:,j])
        #                * kn.cubic_spline_kernel_derivative(X_i, X_j, h))

    # print(particle)
    laplace_i = 2*alpha_i*particle

    return laplace_i


def initial_conditions(lx, ly, npx, npy, T_0y, T_1y, T_x0, T_x1, T_xy):
    x = np.linspace(0, lx, npx)
    y = np.linspace(0, ly, npy)

    np_2d = npx*npy

    x_field, y_field = np.meshgrid(x, y)
    x_field_1d = np.reshape(x_field, (1, np_2d))
    y_field_1d = np.reshape(y_field, (1, np_2d))

    idx_0 = np.where(x_field_1d[0,:,None] == 0.0)[0]
    idx_1 = np.where(x_field_1d[0,:,None] == 1.0)[0]

    idy_0 = np.where(y_field_1d[0,:,None] == 0.0)[0]
    idy_1 = np.where(y_field_1d[0,:,None] == 1.0)[0]

    T_field = np.full((1, np_2d), T_xy)
    T_field[:,idx_0] = T_0y
    T_field[:,idx_1] = T_1y
    T_field[:,idy_0] = T_x0
    T_field[:,idy_1] = T_x1

    # print(idx_0)
    # print(idx_1)

    # print(idy_0)
    # print(idy_1)

    return T_field


def analytical_solution(particle_list, Ts):
    num_series_iter = 10
    num_particles = particle_list.shape[1]
    N = num_particles
    T_exact = np.zeros((1, num_particles))
    delta_T = np.zeros((1, num_particles))

    for i in range(0,num_particles):
        for N in range(1,num_series_iter):
            AN = ((2*Ts)/(N*math.pi))*(((-1)**N - 1)/(np.sinh(N*math.pi)))
            delta_T[0,i] = AN*np.sin(N*math.pi*particle_list[0,i])\
                * np.sinh(N*math.pi*(particle_list[1,i] - 1))
            np.add(T_exact, delta_T, out=T_exact, casting="unsafe")

    return T_exact


def numerical_error(T_numerical, T_analytical):
    return (T_analytical - T_numerical)


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

    T_0y = 0
    T_1y = 0
    T_x0 = Ts
    T_x1 = 0
    T_xy = 0

    dt = 1e-1
    t_final = 1
    time_steps = np.arange(0, t_final, step=dt)
    num_steps = len(time_steps)

    particle_list = particle_creation(Lx, Ly, num_particles_x, num_particles_y)
    num_particles = particle_list.shape[1]
    particle_densities = np.ones((1, num_particles))
    particle_masses = dx*dy*particle_densities
    T_field = initial_conditions(Lx, Ly, num_particles_x, num_particles_y,
                                 T_0y, T_1y, T_x0, T_x1, T_xy)

    # print(T_field)
    print('Solver')
    # num_particles = 1
    laplacian = np.zeros((1, num_particles))
    # print(laplacian)

    for i_time in range(0, num_steps):
        print("Time iteration #{:d}".format(i_time+1))
        for i in range(0, num_particles):
            # print("Particle #{:d}".format(i+1))
            laplacian[0,i] = discretized_laplacian(K, rho, cv, dx, dy,
                                                 particle_masses,
                                                 particle_densities,
                                                 particle_list, T_field, i)
            # print(laplacian[0,i])
        # print(laplacian)
        # print(dt)
        # T_field += (laplacian*dt)
        np.add(T_field, laplacian*dt, out=T_field, casting="unsafe")

    print('Postprocessing')
    # print(T_field)

    T_exact = analytical_solution(particle_list, Ts)
    # print(T_exact)

    interpolation_error = numerical_error(T_field, T_exact)
    print('Maximum temperature difference: {:.3f} °C'
          .format(np.max(interpolation_error)))

    plt.scatter(particle_list[0,:], particle_list[1,:], c=T_exact)
    plt.figure()
    plt.scatter(particle_list[0,:], particle_list[1,:], c=T_field)
    plt.show()



if __name__ == "__main__":
    main()

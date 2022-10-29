import numpy as np
import kernel as kn


def initial_conditions(particle_list, T_0y, T_1y, T_x0, T_x1, T_xy):

    num_particles = particle_list.shape[0]

    idx_0 = np.where(particle_list[:, 0] == 0.0)[0]
    idx_1 = np.where(particle_list[:, 0] == 1.0)[0]

    idy_0 = np.where(particle_list[:, 1] == 0.0)[0]
    idy_1 = np.where(particle_list[:, 1] == 1.0)[0]

    T_field = np.full((num_particles, 1), T_xy, dtype='float64')
    T_field[idy_0, 0] = T_x0  # Ts
    T_field[idx_0, 0] = T_0y
    T_field[idx_1, 0] = T_1y
    T_field[idy_1, 0] = T_x1

    return T_field

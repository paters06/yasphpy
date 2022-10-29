import math
import numpy as np


def analytical_solution(particle_list, Ts):
    num_series_iter = 91
    num_particles = particle_list.shape[0]
    N = num_particles
    T_exact = np.zeros((num_particles, 1))
    delta_T = np.zeros((num_particles, 1))

    for i in range(0, num_particles):
        for N in range(1, num_series_iter):
            AN = ((2*Ts)/(N*math.pi))*(((-1)**N - 1)/(np.sinh(N*math.pi)))
            delta_T[i, 0] = AN*np.sin(N*math.pi*particle_list[i, 0])\
                * np.sinh(N*math.pi*(particle_list[i, 1] - 1))
            T_exact += delta_T

    return T_exact

def numerical_error(T_numerical: np.ndarray, T_analytical: np.ndarray) -> np.ndarray:
    num_cols = T_numerical.shape[1]
    if num_cols == 1:
        return np.abs(T_analytical - T_numerical)
    else:
        max_col = num_cols - 1
        return np.abs(T_analytical - T_numerical[:, max_col])

def postprocessing_fields(points, field):
    points_x = np.unique(points[:, 0])
    points_y = np.unique(points[:, 1])
    num_x = len(points_x)
    num_y = len(points_y)
    field_xy = np.reshape(field, (num_x, num_y))
    return points_x, points_y, field_xy
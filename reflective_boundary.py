import numpy as np
from numpy.linalg import norm


class ReflectiveBoundary:
    def __init__(self, bounding_box: list, reflective_walls: list,
                 CR: float, dt: float) -> None:
        self.Lx = bounding_box[0]
        self.Ly = bounding_box[1]
        self.CR = CR

        self.dt = dt
        self.reflective_walls = reflective_walls

        self.n_top = np.array([0, -1.0])
        self.n_down = np.array([0, 1.0])
        self.n_left = np.array([1, 0])
        self.n_right = np.array([-1, 0])

        self.tol = 1e-2
        # self._print_box_dimensions()

    def _move_individual_particle(self,
                                  coor: np.ndarray, vel: np.ndarray,
                                  dv_dt: float):
        coor_f = coor.copy()
        vel_f = vel.copy()

        vel_f = vel + dv_dt*self.dt
        coor_f = coor + vel_f*self.dt
        return coor_f, vel_f

    def analyze_movement(self, Co: np.ndarray, vo: np.ndarray,
                         radius: float) -> tuple[np.ndarray, np.ndarray]:
        dv_dt = 0.0
        Cf, vf = self._move_individual_particle(Co, vo, dv_dt)

        does_collide, distances_to_edges = self.detect_collision(Co, Cf)

        while does_collide:
            CI, n_vec_list = self.find_contact_point(Co, Cf, distances_to_edges, radius)  # type: ignore
            Cf, vf = self.correct_position(CI, Cf, vf, n_vec_list)
            does_collide, distances_to_edges = self.detect_collision(CI, Cf)
            Co = CI.copy()

        # self._print_corrected_variables(Cf, vf)
        return Cf, vf

    def detect_collision(self, Co: np.ndarray,
                         C1: np.ndarray) -> tuple[bool, list]:
        Xo_top = np.array([Co[0], self.Ly])
        Xo_down = np.array([Co[0], 0.0])
        Xo_left = np.array([0, Co[1]])
        Xo_right = np.array([self.Lx, Co[1]])

        d_top = np.dot(self.n_top, C1 - Xo_top)
        d_down = np.dot(self.n_down, C1 - Xo_down)
        d_left = np.dot(self.n_left, C1 - Xo_left)
        d_right = np.dot(self.n_right, C1 - Xo_right)

        distances_to_edges = [d_top, d_down, d_left, d_right]

        d_min = min(distances_to_edges)

        if d_min < self.tol:
            # print('Collision!!!')
            does_collide = True
        else:
            does_collide = False

        return does_collide, distances_to_edges

    def find_contact_point(self, Co: np.ndarray, C1: np.ndarray,
                           distances_to_edges: list, radius: float):
        Po = np.zeros((2,))
        Xo = np.zeros((2,))
        n_vec = np.zeros((2,))
        n_vec_list = []

        CI = C1.copy()

        d_min = min(distances_to_edges)

        d_top = distances_to_edges[0]
        d_down = distances_to_edges[1]
        d_left = distances_to_edges[2]
        d_right = distances_to_edges[3]

        t_vec_norm = norm(C1 - Co)
        if t_vec_norm > 1e-5:
            t_vec = (C1 - Co)/t_vec_norm
        else:
            t_vec = (C1 - Co)

        if d_min < self.tol:
            if (d_top * d_down) < self.tol:
                if d_top < self.tol:
                    Po = Co + np.array([0, radius])
                    Xo = np.array([Co[0], self.Ly])
                    n_vec = self.n_top
                    n_vec_list.append(self.n_top)
                if d_down < self.tol:
                    Po = Co - np.array([0, radius])
                    Xo = np.array([Co[0], 0.0])
                    n_vec = self.n_down
                    n_vec_list.append(self.n_down)

            if (d_left * d_right) < self.tol:
                if d_right < self.tol:
                    Po = Co + np.array([radius, 0])
                    Xo = np.array([self.Lx, Co[1]])
                    n_vec = self.n_right
                    n_vec_list.append(self.n_right)
                if d_left < self.tol:
                    Po = Co - np.array([radius, 0])
                    Xo = np.array([0, Co[1]])
                    n_vec = self.n_left
                    n_vec_list.append(self.n_left)

            s_parameter = np.dot(Xo - Po, n_vec)/np.dot(t_vec, n_vec)
            PI = Po + s_parameter*t_vec
            CI = Co + s_parameter*t_vec

        return CI, n_vec_list

    def correct_position(self, CI: np.ndarray, C1: np.ndarray,
                         vo: np.ndarray, n_vec_list: list):
        Cf = C1.copy()
        vf = vo.copy()

        for i in range(len(n_vec_list)):
            a_i = abs(np.dot(C1 - CI, n_vec_list[i]))
            if abs(n_vec_list[i][0]) < self.tol:
                k_col = 1
            else:
                k_col = 0

            vf[k_col] = -self.CR*vo[k_col]
            Cf = Cf + (1.0 + self.CR)*(a_i)*n_vec_list[i]

        return Cf, vf

    def _print_box_dimensions(self):
        print(self.Lx)
        print(self.Ly)

    def _print_corrected_variables(self, Cf: np.ndarray, vf: np.ndarray):
        print('------------')
        print('x: ', Cf)
        print('------------')
        print('vx: ', vf)
        print('------------')


def main():
    bounding_box = [0.5, 0.5]
    CR = 1.0
    dt = 1.0

    Co = np.array([1.01, 1.01])
    vo = np.array([1.0, 1.0])
    radius = 0.01

    boundary = ReflectiveBoundary(bounding_box, CR, dt)
    boundary.analyze_movement(Co, vo, radius)


if __name__ == '__main__':
    main()

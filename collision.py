import numpy as np
from numpy.linalg import norm


class Point:
    def __init__(self, radius, x, y, vx, vy, dt) -> None:
        self.radius = radius
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.dt = dt

    def update_variables(self, dv_dt):
        # Update velocities
        self.velocity += dv_dt*self.dt

        # Update positions
        self.position += self.velocity*self.dt

    def get_coordinates(self):
        return self.position

    def get_velocities(self):
        return self.velocity

    def set_coordinates(self, val_coor):
        """
        BEWARE THIS FUNCTION
        NUMPY ARRAYS ARE PASSED BY REFERENCE!!!
        """
        self.position = val_coor

    def set_velocities(self, val_vel):
        self.velocity = val_vel

    def print_variables(self):
        print('************')
        self._print_positions()
        # self._print_velocities()
        print('************')

    def _print_positions(self):
        print('------------')
        print('x: ', self.position)
        print('------------')

    def _print_velocities(self):
        print('------------')
        print('vx: ', self.velocity)
        print('------------')


class Collision:
    def __init__(self, pt: Point, CR: float) -> None:
        self.Lx = 0.5
        self.Ly = 0.5
        self.CR = CR

        self.to = 0.0
        self.dt = 1.0

        self.n_top = np.array([0, -1.0])
        self.n_down = np.array([0, 1.0])
        self.n_left = np.array([1, 0])
        self.n_right = np.array([-1, 0])

        self.pt = pt
        self.point_path = []
        self.tol = 1e-2
        # self._print_box_dimensions()

    def move_point(self, tf):
        ti = self.dt
        Co = self.pt.get_coordinates().copy()
        while ti <= tf:
            # self._print_time(ti)
            Co = self.pt.get_coordinates().copy()
            self.pt.update_variables(0.0)
            Cf = self.pt.get_coordinates().copy()

            vo = self.pt.get_velocities().copy()
            vf = vo.copy()

            does_collide, distances_to_edges = self.detect_collision(Co, Cf)

            while does_collide:
                CI, n_vec_list = self.find_contact_point(Co, Cf, distances_to_edges)  # type: ignore
                Cf, vf = self.correct_position(CI, Cf, vf, n_vec_list)
                does_collide, distances_to_edges = self.detect_collision(CI, Cf)  # type: ignore
                Co = CI.copy()

            self.point_path.append([ti, Cf.copy(), vf.copy()])
            self.pt.set_coordinates(Cf)
            self.pt.set_velocities(vf)
            ti += self.dt

        self._print_point_path()

    def detect_collision(self, Co, C1) -> tuple[bool, list]:
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

    def find_contact_point(self, Co, C1, distances_to_edges):
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
                    Po = Co + np.array([0, self.pt.radius])
                    Xo = np.array([Co[0], self.Ly])
                    n_vec = self.n_top
                    n_vec_list.append(self.n_top)
                if d_down < self.tol:
                    Po = Co - np.array([0, self.pt.radius])
                    Xo = np.array([Co[0], 0.0])
                    n_vec = self.n_down
                    n_vec_list.append(self.n_down)

            if (d_left * d_right) < self.tol:
                if d_right < self.tol:
                    Po = Co + np.array([self.pt.radius, 0])
                    Xo = np.array([self.Lx, Co[1]])
                    n_vec = self.n_right
                    n_vec_list.append(self.n_right)
                if d_left < self.tol:
                    Po = Co - np.array([self.pt.radius, 0])
                    Xo = np.array([0, Co[1]])
                    n_vec = self.n_left
                    n_vec_list.append(self.n_left)

            s_parameter = np.dot(Xo - Po, n_vec)/np.dot(t_vec, n_vec)
            PI = Po + s_parameter*t_vec
            CI = Co + s_parameter*t_vec

        return CI, n_vec_list

    def correct_position(self, CI, C1, vo, n_vec_list):
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

    def _print_time(self, current_time):
        print('t: ', current_time, ' s')

    def _print_point_path(self):
        for pt in self.point_path:
            print(*pt)


def test_01():
    """FULL SUCCESS"""
    print("TEST 01")
    pt = Point(0.01, 0.01, 0.01, 1.0, 1.0, 1.0)

    box = Collision(pt, 1.0)
    box.move_point(8)


def test_02():
    """FULL SUCCESS"""
    print("TEST 02")
    pt = Point(0.01, 0.25, 0.25, 1.0, 0.0, 1.0)

    box = Collision(pt, 1.0)
    box.move_point(8)


def test_03():
    """FAIL"""
    print("TEST 03")
    pt = Point(0.01, 0.25, 0.31, 0.0, 1.0, 1.0)

    box = Collision(pt, 1.0)
    box.move_point(8)


def test_04():
    """FULL SUCCESS"""
    print("TEST 04")
    pt = Point(0.01, 0.01, 0.01, 0.87, 0.5, 1.0)

    box = Collision(pt, 1.0)
    box.move_point(8)


def test_05():
    """FULL SUCESS"""
    print("TEST 05")
    pt = Point(0.01, 0.01, 0.01, 1.0, 1.0, 1.0)

    box = Collision(pt, 0.9)
    box.move_point(8)


def test_06():
    """FULL SUCESS"""
    print("TEST 06")
    pt = Point(0.01, 0.25, 0.25, 1.0, 0.0, 1.0)

    box = Collision(pt, 0.9)
    box.move_point(8)


def test_07():
    """FULL SUCCESS"""
    print("TEST 07")
    pt = Point(0.01, 0.25, 0.25, 0.0, 1.0, 1.0)

    box = Collision(pt, 0.9)
    box.move_point(8)


def test_08():
    """FULL SUCESS"""
    print("TEST 08")
    pt = Point(0.01, 0.01, 0.01, 0.87, 0.5, 1.0)

    box = Collision(pt, 0.9)
    box.move_point(8)


def test_09():
    """FULL SUCCESS"""
    print("TEST 09")
    pt = Point(0.01, 0.01, 0.01, 1.0, 1.0, 1.0)

    box = Collision(pt, 0.0)
    box.move_point(3)


def test_10():
    """FULL SUCCESS"""
    print("TEST 10")
    pt = Point(0.01, 0.25, 0.25, 1.0, 0.0, 1.0)

    box = Collision(pt, 0.0)
    box.move_point(3)


def main():
    test_01()


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    main()

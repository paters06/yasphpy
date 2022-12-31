import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def progress_callback(current_frame: int, total_frames: int):
    print(f'Saving frame {current_frame}/{total_frames}')


class CloudPoint:
    def __init__(self, input_type: str,
                 num_points: int, 
                 radius: float, dt: float) -> None:
        self.dt = dt
        self.num_points = num_points
        self.radius = radius

        if input_type == "random":
            self._create_random_points()

    def _create_random_points(self):
        self.points_positions = np.random.rand(self.num_points, 2)
        self.points_velocities = np.random.rand(self.num_points, 2)
        self.points_radii = self.radius*np.ones((self.num_points, 1))

    def set_positions(self, points_positions: np.ndarray):
        self.points_positions = points_positions

    def set_velocities(self, points_velocities: np.ndarray):
        self.points_velocities = points_velocities

    def set_radii(self, points_radii: np.ndarray):
        self.points_radii = points_radii

    def get_num_points(self):
        return self.num_points

    def get_positions(self):
        return self.points_positions

    def get_velocitites(self):
        return self.points_velocities

    def get_radii(self):
        return self.points_radii

    def update_variables(self, dv_dt):
        for j in range(self.num_points):
            # Update velocities
            self.points_velocities[j, :] += dv_dt*self.dt

            # Update positions
            self.points_positions[j, :] += self.points_velocities[j, :]*self.dt

    def _print_info_cloud_points(self):
        print('Number of points: {}'.format(self.num_points))
        print('Positions')
        print(self.points_positions)
        print('Velocities')
        print(self.points_velocities)


class CollisionNPoints:
    def __init__(self, bounding_box: list, cloud_pt: CloudPoint,
                 tf: float, CR: float, dt: float) -> None:
        self.Lx = bounding_box[0]
        self.Ly = bounding_box[1]
        self.CR = CR

        self.to = 0.0
        self.dt = dt

        self.n_top = np.array([0, -1.0])
        self.n_down = np.array([0, 1.0])
        self.n_left = np.array([1, 0])
        self.n_right = np.array([-1, 0])

        self.num_points = cloud_pt.get_num_points()
        self.positions = cloud_pt.get_positions().copy()
        self.velocities = cloud_pt.get_velocitites().copy()
        self.radii = cloud_pt.get_radii()
        self.tol = 1e-2

        self.initial_positions = cloud_pt.get_positions().copy()
        self.initial_velocities = cloud_pt.get_velocitites().copy()

        self.time_steps = np.arange(1.0, tf + self.dt, step=self.dt)
        self.num_time_steps = len(self.time_steps)

        self.positions_list = np.zeros((self.num_points, 2*self.num_time_steps))
        self.velocities_list = np.zeros((self.num_points, 2*self.num_time_steps))

        self.move_points()

        # self._print_variable_list()
        self._plot_cloud()

    def _move_individual_particle(self,
                                  coor: np.ndarray,
                                  vel: np.ndarray,
                                  dv_dt: float):
        coor_f = coor.copy()
        vel_f = vel.copy()

        vel_f = vel + dv_dt*self.dt
        coor_f = coor + vel_f*self.dt
        return coor_f, vel_f

    def move_points(self):
        dv_dt = 0.0
        for i in range(0, self.num_time_steps):
            for j in range(0, self.num_points):
                # self._print_time(ti)
                Co = self.positions[j, :].copy()
                vo = self.velocities[j, :].copy()
                r_j = self.radii[j, 0].copy()

                Cf, vf = self._move_individual_particle(Co, vo, dv_dt)
                C1 = Cf.copy()

                does_collide, d_list, collision_side, CI = self.detect_collision(Co, C1, r_j)

                while does_collide:
                    Cf, vf = self.correct_position(Cf, vf, d_list, collision_side, r_j)
                    does_collide, d_list, collision_side, CI = self.detect_collision(CI, Cf, r_j)

                self._assemble_variables_values(i, j, Cf.copy(), vf.copy())

                self.positions[j, :] = Cf.copy()
                self.velocities[j, :] = vf.copy()

    def detect_collision(self, Co, C1, radius):
        Xo_top = np.array([Co[0], self.Ly])
        Xo_down = np.array([Co[0], 0.0])
        Xo_left = np.array([0, Co[1]])
        Xo_right = np.array([self.Lx, Co[1]])

        d_top = norm(C1 - Xo_top)
        d_down = norm(C1 - Xo_down)
        d_left = norm(C1 - Xo_left)
        d_right = norm(C1 - Xo_right)

        d_top = np.dot(self.n_top, C1 - Xo_top)
        d_down = np.dot(self.n_down, C1 - Xo_down)
        d_left = np.dot(self.n_left, C1 - Xo_left)
        d_right = np.dot(self.n_right, C1 - Xo_right)

        t_vec_norm = norm(C1 - Co)
        if t_vec_norm > 1e-5:
            t_vec = (C1 - Co)/t_vec_norm
        else:
            t_vec = (C1 - Co)

        d_min = min((d_top, d_down, d_left, d_right))  # type: ignore

        collision_side = []
        d_list = []
        Po = np.zeros((2,))
        Xo = np.zeros((2,))
        n_vec = np.zeros((2,))

        CI = C1.copy()

        if d_min < self.tol:
            if (d_top * d_down) < self.tol:
                if d_top < self.tol:
                    collision_side.append('top')
                    d_list.append(d_top)
                    Po = Co + np.array([0, radius])
                    Xo = Xo_top
                    n_vec = self.n_top
                if d_down < self.tol:
                    collision_side.append('down')
                    d_list.append(d_down)
                    Po = Co - np.array([0, radius])
                    Xo = Xo_down
                    n_vec = self.n_down

            if (d_left * d_right) < self.tol:
                if d_right < self.tol:
                    collision_side.append('right')
                    d_list.append(d_right)
                    Po = Co + np.array([radius, 0])
                    Xo = Xo_right
                    n_vec = self.n_right
                if d_left < self.tol:
                    collision_side.append('left')
                    d_list.append(d_left)
                    Po = Co - np.array([radius, 0])
                    Xo = Xo_left
                    n_vec = self.n_left

            s_parameter = np.dot(Xo - Po, n_vec)/np.dot(t_vec, n_vec)
            PI = Po + s_parameter*t_vec
            CI = Co + s_parameter*t_vec

        if d_min < self.tol:
            # print('Collision!!!')
            does_collide = True
        else:
            does_collide = False

        return does_collide, d_list, collision_side, CI

    def correct_position(self, C1, vo, d_list, collision_side, radius):
        Cf = C1.copy()
        vf = vo.copy()
        if len(collision_side) == 2:
            k_col = 0
            for i in range(0, 2):
                if collision_side[i] == 'right':
                    M_vec = self.n_right
                    k_col = 0
                elif collision_side[i] == 'left':
                    M_vec = self.n_left
                    k_col = 0
                elif collision_side[i] == 'top':
                    M_vec = self.n_top
                    k_col = 1
                elif collision_side[i] == 'down':
                    M_vec = self.n_down
                    k_col = 1
                else:
                    M_vec = np.zeros((2,))
                vf[k_col] = -self.CR*vo[k_col]
                Cf = Cf + (1.0 + self.CR)*(radius - d_list[i])*M_vec
        elif len(collision_side) == 1:
            if collision_side[0] == 'right':
                vf[0] = -self.CR*vo[0]
                M_vec = self.n_right
            elif collision_side[0] == 'left':
                vf[0] = -self.CR*vo[0]
                M_vec = self.n_left
            elif collision_side[0] == 'top':
                vf[1] = -self.CR*vo[1]
                M_vec = self.n_top
            elif collision_side[0] == 'down':
                vf[1] = -self.CR*vo[1]
                M_vec = self.n_down
            else:
                M_vec = np.zeros((2,))

            Cf = Cf + (1.0 + self.CR)*(radius - d_list[0])*M_vec
        else:
            pass
        return Cf, vf

    def _print_box_dimensions(self):
        print(self.Lx)
        print(self.Ly)

    def _print_time(self, current_time):
        print('t: ', current_time, ' s')

    def _assemble_variables_values(self, i, j, Cf, vf):
        self.positions_list[j, 2*i] = Cf[0]
        self.positions_list[j, 2*i+1] = Cf[1]

        self.velocities_list[j, 2*i] = vf[0]
        self.velocities_list[j, 2*i+1] = vf[1]

    def _print_variable_list(self):
        print('Positions')
        print(self.positions_list)
        print('Velocities')
        print(self.velocities_list)

    def _plot_cloud(self):
        fig, ax = plt.subplots()
        pt_o = self.initial_positions
        im = ax.plot(pt_o[:, 0], pt_o[:, 1], '.', c='k')

        pt_i = self.positions_list

        ims = [im]

        num_images = self.num_time_steps

        for i_img in range(num_images):
            im = ax.plot(pt_i[:, 2*i_img], pt_i[:, 2*i_img+1], '.', c='k')
            ims.append(im)

        ani = animation.ArtistAnimation(fig, ims, blit=True)

        writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Me'),
                                        bitrate=1800)

        dirname = '../yasphpy_images/'
        filename = 'particles_colliding_random_input.mp4'

        ani.save(dirname+filename, writer=writer,
                 progress_callback=progress_callback)


def test_manual_input():
    points_positions = np.array([[0.01, 0.01],
                                 [0.25, 0.25],
                                 [0.01, 0.01]])
    points_velocities = np.array([[1.0, 1.0],
                                  [1.0, 0.0],
                                  [0.87, 0.5]])
    radius = 0.01
    num_points = points_positions.shape[0]
    points_radii = radius*np.ones((num_points, 1))
    dt = 0.01

    cloud = CloudPoint("manual", num_points, radius, dt)
    cloud.set_positions(points_positions)
    cloud.set_velocities(points_velocities)
    cloud.set_radii(points_radii)
    # cloud._print_info_cloud_points()

    tf = 8.0
    CR = 1.0
    bounding_box = [0.5, 0.5]
    collision = CollisionNPoints(bounding_box, cloud, tf, CR, dt)


def test_random_input():
    radius = 0.01
    num_points = 300
    dt = 0.01

    cloud = CloudPoint("random", num_points, radius, dt)

    tf = 8.0
    CR = 1.0
    bounding_box = [1.0, 1.0]
    collision = CollisionNPoints(bounding_box, cloud, tf, CR, dt)


def main():
    # test_manual_input()
    test_random_input()


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    main()

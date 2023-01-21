import numpy as np


class Particles:

    def __init__(self, Lx, Ly, np_x: int, np_y: int, rho=1.0):
        """
        Class constructor

        Input:
        -----------
            - Lx: length in x direction
            - Ly: length in y direction
            - np_x: number of particle on x side
            - np_y: number of particle on y side
            - rho: density of the medium
        """
        self.Lx = Lx
        self.Ly = Ly
        self.np_x = np_x
        self.np_y = np_y
        self.dx = Lx/np_x
        self.dy = Ly/np_y
        self.rho = rho

        self.create_particles()
        self.compute_masses()

    def create_particles(self) -> None:
        self._create_domain_particles()
        self._create_boundary_particles()
        self._create_virtual_particles_type_II()
        # self.particle_list = np.vstack((self.boundary_particles,
        #                                 self.domain_particles))
        self.particle_list = self.domain_particles.copy()
        self.particle_velocities = np.zeros_like(self.particle_list)

    def compute_masses(self) -> None:
        num_particles = self.particle_list.shape[0]
        self.particle_densities = self.rho*np.ones((num_particles, 1))
        self.particle_masses = (self.dx*self.dy)*self.particle_densities

    def _create_domain_particles(self) -> None:
        x = np.linspace(0.0, self.Lx, self.np_x)
        y = np.linspace(0.0, self.Ly, self.np_y)

        num_x = len(x)
        num_y = len(y)

        x_dom = x[1:num_x-1]
        y_dom = y[1:num_y-1]

        np_2d = (self.np_x - 2)*(self.np_y - 2)

        x_field, y_field = np.meshgrid(x_dom, y_dom)
        x_field_1d = np.reshape(x_field, (np_2d, 1))
        y_field_1d = np.reshape(y_field, (np_2d, 1))

        self.domain_particles = np.hstack((x_field_1d, y_field_1d))

    def _create_boundary_particles(self) -> None:
        """
        Note: For the case of virtual particles,
        the code used twice the number of particles
        on each side of the geometry
        """
        x = np.linspace(0.0, self.Lx, self.np_x)
        y = np.linspace(0.0, self.Ly, self.np_y)

        np_2d = (self.np_x)*(self.np_y)

        x_field, y_field = np.meshgrid(x, y)
        x_field_1d = np.reshape(x_field, (np_2d, 1))
        y_field_1d = np.reshape(y_field, (np_2d, 1))

        idx_0 = np.where(x_field_1d == 0.0)[0]
        idx_1 = np.where(x_field_1d == self.Lx)[0]

        idy_0 = np.where(y_field_1d == 0.0)[0]
        idy_1 = np.where(y_field_1d == self.Ly)[0]

        id_bounds = np.hstack((idx_0, idx_1, idy_0, idy_1))
        id_boundary = np.unique(id_bounds)

        x_boundary1d = x_field_1d[id_boundary]
        y_boundary1d = y_field_1d[id_boundary]

        self.boundary_particles = np.hstack((x_boundary1d, y_boundary1d))

    def _create_virtual_particles_type_II(self) -> None:
        x_min = -0.1*self.Lx
        x_max = 1.1*self.Lx

        y_min = -0.1*self.Ly
        y_max = 1.1*self.Ly

        nx = self.np_x + 4
        ny = self.np_y + 4

        np_2d = (nx)*(ny)

        # Bottom band
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        x_2d, y_2d = np.meshgrid(x, y)
        x_1d = np.reshape(x_2d, (np_2d, 1))
        y_1d = np.reshape(y_2d, (np_2d, 1))

        idx_virtual1 = np.where(x_1d < 0.0)[0]
        idx_virtual2 = np.where(x_1d > self.Lx)[0]
        idx_virtual = np.union1d(idx_virtual1, idx_virtual2)

        idy_virtual1 = np.where(y_1d < 0.0)[0]
        idy_virtual2 = np.where(y_1d > self.Ly)[0]
        idy_virtual = np.union1d(idy_virtual1, idy_virtual2)

        # id_virtual = np.unique((idx_virtual, idy_virtual))
        id_virtual = np.union1d(idx_virtual, idy_virtual)

        x_virtual = x_1d[id_virtual]
        y_virtual = y_1d[id_virtual]

        self.virtual_particles = np.hstack((x_virtual, y_virtual))

    def get_dx(self):
        return self.dx

    def get_dy(self):
        return self.dy

    def get_particle_list(self):
        return self.particle_list

    def get_domain_particle_list(self):
        return self.domain_particles

    def get_virtual_particles_list(self):
        return self.virtual_particles

    def get_num_particles(self):
        return self.particle_list.shape[0]

    def get_num_domain_particles(self):
        return self.domain_particles.shape[0]

    def get_num_virtual_particles(self):
        return self.virtual_particles.shape[0]

    def get_num_boundary_particles(self):
        return self.boundary_particles.shape[0]

    def get_particle_velocities(self):
        return self.particle_velocities

    def get_particle_densities(self):
        return self.particle_densities

    def get_particle_masses(self):
        return self.particle_masses

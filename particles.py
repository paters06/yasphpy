import numpy as np


class Particles:
    def __init__(self) -> None:
        pass

    def __init__(self, Lx, Ly, np_side, rho=1.0):
        """
        Class constructor

        Input:
            - Lx: length in x direction
            - Ly: length in y direction
            - np_side: number of particle per side
            - rho: density of the medium
        """
        self.Lx = Lx
        self.Ly = Ly
        self.np_side = np_side
        self.dx = Lx/np_side
        self.dy = Ly/np_side
        self.rho = rho

    def create_particles(self) -> None:
        self._create_domain_particles()
        self._create_boundary_particles()
        self.particle_list = np.hstack((self.boundary_particles,
                                        self.domain_particles))

    def compute_masses(self) -> None:
        num_particles = self.particle_list.shape[1]
        self.particle_densities = self.rho*np.ones((1, num_particles))
        self.particle_masses = (self.dx*self.dy)*self.particle_densities

    def _create_domain_particles(self) -> None:
        x = np.linspace(0.0, self.Lx, self.np_side)
        y = np.linspace(0.0, self.Ly, self.np_side)

        x_dom = x[1:-1]
        y_dom = y[1:-1]

        np_2d = (self.np_side - 2)*(self.np_side - 2)

        x_field, y_field = np.meshgrid(x_dom, y_dom)
        x_field_1d = np.reshape(x_field, (1, np_2d))
        y_field_1d = np.reshape(y_field, (1, np_2d))

        self.domain_particles = np.vstack((x_field_1d, y_field_1d))

    def _create_boundary_particles(self) -> None:
        x = np.linspace(0.0, self.Lx, 2*self.np_side)
        y = np.linspace(0.0, self.Ly, 2*self.np_side)

        np_2d = 4*(self.np_side)*(self.np_side)

        x_field, y_field = np.meshgrid(x, y)
        x_field_1d = np.reshape(x_field, (1, np_2d))
        y_field_1d = np.reshape(y_field, (1, np_2d))

        idx_0 = np.where(x_field_1d[0, :, None] == 0.0)[0]
        idx_1 = np.where(x_field_1d[0, :, None] == self.Lx)[0]

        idy_0 = np.where(y_field_1d[0, :, None] == 0.0)[0]
        idy_1 = np.where(y_field_1d[0, :, None] == self.Ly)[0]

        id_bounds = np.stack((idx_0, idx_1, idy_0, idy_1))
        id_boundary = np.unique(id_bounds)

        x_boundary1d = x_field_1d[0, id_boundary]
        y_boundary1d = y_field_1d[0, id_boundary]

        self.boundary_particles = np.vstack((x_boundary1d, y_boundary1d))

    def get_dx(self):
        return self.dx

    def get_dy(self):
        return self.dy

    def get_particle_list(self):
        return self.particle_list

    def get_num_particles(self):
        return self.particle_list.shape[1]

    def get_num_domain_particles(self):
        return self.domain_particles.shape[1]

    def get_num_boundary_particles(self):
        return self.boundary_particles.shape[1]

    def get_particle_densities(self):
        return self.particle_densities

    def get_particle_masses(self):
        return self.particle_masses

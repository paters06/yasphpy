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
        self._create_virtual_particles_type_I()
        self._create_virtual_particles_type_II()
        self.particle_list = self.domain_particles.copy()
        self.particle_velocities = np.zeros_like(self.particle_list)
        self.virtual_velocities = np.zeros((self.num_virtual_particles_I, 2))

    def compute_masses(self) -> None:
        num_particles = self.particle_list.shape[0]
        self.particle_densities = self.rho*np.ones((num_particles, 1))
        self.virtual_densities = self.rho*np.ones((self.num_virtual_particles_I, 1))
        self.particle_masses = (self.dx*self.dy)*self.particle_densities
        self.virtual_masses = (self.dx*self.dy)*self.virtual_densities

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

    def _create_virtual_particles_type_I(self) -> None:
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

        self.virtual_particles_I = np.hstack((x_boundary1d, y_boundary1d))
        self.num_virtual_particles_I = self.virtual_particles_I.shape[0]

    def _create_virtual_particles_type_II(self) -> None:

        num_domain_particles = self.domain_particles.shape[0]

        virtual_particles_II = []
        virtual_indices = []
        
        for j in range(num_domain_particles):
            x = self.domain_particles[j, 0]
            y = self.domain_particles[j, 1]

            if abs(x) <= (2.1*self.dx):
                virtual_pt_II_j = [-x, y]
                virtual_particles_II.append(virtual_pt_II_j)
            
            if abs(x - self.Lx) < (2.1*self.dx):
                virtual_pt_II_j = [2*self.Lx - x, y]
                virtual_particles_II.append(virtual_pt_II_j)
            
            if abs(y) < (2.1*self.dy):
                virtual_pt_II_j = [x, -y]
                virtual_particles_II.append(virtual_pt_II_j)
            
            if abs(y - self.Ly) < (2.1*self.dy):
                virtual_pt_II_j = [x, 2*self.Ly - y]
                virtual_particles_II.append(virtual_pt_II_j)

            # virtual_particles_II.append(virtual_pt_II_j)

        self.virtual_particles_II = np.array(virtual_particles_II)
        self.num_virtual_particles_II = self.virtual_particles_II.shape[0]

    def get_dx(self):
        return self.dx

    def get_dy(self):
        return self.dy

    def get_domain_particles(self):
        return self.domain_particles

    def get_virtual_particles_I(self):
        return self.virtual_particles_I

    def get_virtual_particles_II(self):
        return self.virtual_particles_II

    def get_num_domain_particles(self):
        return self.domain_particles.shape[0]

    def get_num_virtual_particles_I(self):
        return self.num_virtual_particles_I

    def get_num_virtual_particles_II(self):
        return self.num_virtual_particles_II

    def get_particle_velocities(self):
        return self.particle_velocities

    def get_particle_densities(self):
        return self.particle_densities

    def get_particle_masses(self):
        return self.particle_masses

    def get_virtual_velocities(self):
        return self.virtual_velocities

    def get_virtual_densities(self):
        return self.virtual_densities

    def get_virtual_masses(self):
        return self.virtual_masses

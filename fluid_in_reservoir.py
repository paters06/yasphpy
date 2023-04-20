from fluid_hydrostatics import FluidHydrostatics
from particles import Particles


def main():
    num_particles_side = 20
    Lx = 1.0
    Ly = 1.0
    nu = 1e-6
    H = Ly
    part1 = Particles(Lx, Ly, num_particles_side)
    reservoir = FluidHydrostatics(part1, nu, H)
    reservoir.update_densities_velocities(part1)


if __name__ == '__main__':
    main()

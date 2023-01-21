from incompressible_flow import IncompressibleFlow


def main():
    num_particles_y = 10
    L = 10.0
    D = 1.0
    rho = 1000.0
    nu = 1e-6
    v_max = 1.0
    P_inlet = 200000.0
    P_outlet = 101325.0
    steady_flow = IncompressibleFlow(L, D, v_max, nu, rho, P_inlet, P_outlet)
    # steady_flow.solve_fluid_equations()


if __name__ == '__main__':
    main()

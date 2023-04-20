from incompressible_flow import IncompressibleFlow
from profiling_script import profiling_script


def main():
    L = 1.0
    D = 1.0
    rho = 1000.0
    nu = 1e-6
    v_max = 0.001
    Re = (v_max*L)/nu
    print('Reynolds: {:.3f}'.format(Re))
    steady_flow = IncompressibleFlow(L, D, v_max, nu, rho)
    # steady_flow.solve_fluid_equations()


if __name__ == '__main__':
    main()
    # profiling_script(main)

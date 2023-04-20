from heat_conduction import HeatConduction
from profiling_script import profiling_script


def main():
    Lx = 1.0
    Ly = 1.0
    num_particles_side = 30

    K = 1.0
    rho = 1.0
    cv = 1.0

    Ts = 100.0

    T_x0 = Ts

    T_0y = 0.0
    T_1y = 0.0

    T_x1 = 0.0
    T_xy = 0.0

    dt = 1.0e-3
    t_final = 1.0
    time_info = [dt, t_final]

    delta_stop = 1e-3

    heat_field = HeatConduction(is_steady=True)

    field_particles = heat_field.discretize_domain(Lx, Ly, num_particles_side)

    heat_field.define_initial_conditions(T_0y, T_1y, T_x0, T_x1, T_xy)

    heat_field.calculate_time_integration(field_particles, time_info,
                                          K, rho, cv, delta_stop)

    heat_field.export_steady_results(show=False,
                                     exact_solution_available=True)

    # heat_field.export_transient_results()


if __name__ == "__main__":
    main()
    # profiling_script(main)

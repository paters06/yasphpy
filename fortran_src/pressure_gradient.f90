subroutine pressure_gradient(h, X_i, P_i, rho_i, num_neighbours, indices, distances, &
                             num_particles, position_list, mass_list, density_list, pressure_list, &
                             grad_P)
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_neighbours, num_particles
    real, dimension(2), intent(in) :: X_i
    real, intent(in) :: P_i
    real, intent(in) :: rho_i
    integer, dimension(num_particles), intent(in) :: indices
    real, dimension(num_particles), intent(in) :: distances
    real, dimension(num_particles), intent(in) :: mass_list
    real, dimension(num_particles, 2), intent(in) :: position_list
    real, dimension(num_particles), intent(in) :: density_list
    real, dimension(num_particles), intent(in) :: pressure_list

    real, dimension(2), intent(out) :: grad_P

    integer :: j, neigh_idx
    real :: r_ij, mass_j, P_j, W, dWdq

    real, dimension(2) :: X_j, rho_j, grad_W

    grad_P = (/0., 0./)

    neighbour_loop: do j = 1, num_neighbours
        neigh_idx = indices(j)
        r_ij = distances(j)
        mass_j = mass_list(neigh_idx)

        nonzero: if (r_ij > 1e-5) then
            X_j = position_list(neigh_idx, :)
            P_j = pressure_list(neigh_idx)
            rho_j = density_list(neigh_idx)

            call cubic_kernel(h, X_i, X_j, W, dWdq, grad_W)

            grad_P = grad_P + mass_j * ((P_i/rho_i**2) + (P_j/rho_j**2))*grad_W
        end if nonzero

    end do neighbour_loop
end subroutine pressure_gradient
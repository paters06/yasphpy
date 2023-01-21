subroutine mass_conservation(h, X_i, vel_i, num_neighbours, indices, distances, &
                              num_particles, mass_list, position_list, velocity_list, &
                              drho_dt)
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_neighbours, num_particles
    real, dimension(2), intent(in) :: X_i
    real, dimension(2), intent(in) :: vel_i
    integer, dimension(num_particles), intent(in) :: indices
    real, dimension(num_particles), intent(in) :: distances
    real, dimension(num_particles), intent(in) :: mass_list
    real, dimension(num_particles, 2), intent(in) :: position_list
    real, dimension(num_particles, 2), intent(in) :: velocity_list
    real, intent(out) :: drho_dt

    integer :: j, neigh_idx
    real :: r_ij, mass_j, W, dWdq

    real, dimension(2) :: X_j, vel_j, grad_W

    drho_dt = 0.0

    neighbour_loop: do j = 1, num_neighbours
        neigh_idx = indices(j)
        r_ij = distances(j)
        X_j = position_list(neigh_idx, :)
        mass_j = mass_list(neigh_idx)
        vel_j = velocity_list(neigh_idx, :)

        nonzero: if (r_ij > 1e-5) then
            call cubic_kernel(h, X_i, X_j, W, dWdq, grad_W)
            drho_dt = drho_dt - mass_j*dot_product((vel_i - vel_j), grad_W)
        end if nonzero

    end do neighbour_loop

end subroutine mass_conservation
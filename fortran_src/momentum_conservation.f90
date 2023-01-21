subroutine momentum_conservation(h, X_i, nu, num_neighbours, indices, distances, &
                                 num_particles, mass_list, density_list, position_list, &
                                 dv_dt)
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_neighbours, num_particles
    real, intent(in) :: nu
    real, dimension(2), intent(in) :: X_i
    integer, dimension(num_particles), intent(in) :: indices
    real, dimension(num_particles), intent(in) :: distances
    real, dimension(num_particles), intent(in) :: mass_list
    real, dimension(num_particles), intent(in) :: density_list
    real, dimension(num_particles, 2), intent(in) :: position_list

    real, dimension(2), intent(out) :: dv_dt

    integer :: j, neigh_idx
    real :: r_ij, mass_j, dist_ij, W, dWdq

    real, dimension(2) :: X_j, rho_j, grad_W, viscous_term

    dv_dt = (/0., 0./)

    viscous_term = (/0., 0./)

    neighbour_loop: do j = 1, num_neighbours
        neigh_idx = indices(j)
        r_ij = distances(j)
        mass_j = mass_list(neigh_idx)
        rho_j = density_list(neigh_idx)
        X_j = position_list(neigh_idx, :)
        dist_ij = sqrt((X_i(1) - X_j(1))**2 + (X_i(2) - X_j(2))**2)

        nonzero: if (dist_ij > 1e-5) then
            call cubic_kernel(h, X_i, X_j, W, dWdq, grad_W)
            viscous_term = viscous_term + &
            2.0*nu*(mass_j/rho_j)*((X_i - X_j)/dist_ij)*grad_W
        end if nonzero

    end do neighbour_loop

    ! fs_i = self.compute_surface_forces(i, distances, indices, num_neighbours, h, dx, dy)

    dv_dt = viscous_term
    !dv_dt = pressure_gradient + viscous_forces + fs_i

end subroutine momentum_conservation
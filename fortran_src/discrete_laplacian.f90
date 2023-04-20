subroutine discrete_laplacian(alpha_i, h, num_particles, m_list, rho_list, &
                              indices, distances, num_neighbours, T_field, T_i, &
                              laplace_i)
    use smoothing_functions
    implicit none
    real, intent(in) :: alpha_i
    real, intent(in) :: h
    integer, intent(in) :: num_particles
    real, dimension(num_particles), intent(in) :: m_list
    real, dimension(num_particles), intent(in) :: rho_list
    integer, dimension(num_particles), intent(in) :: indices
    real, dimension(num_particles), intent(in) :: distances
    integer, intent(in) :: num_neighbours
    real, dimension(num_particles), intent(in) :: T_field
    real, intent(in) :: T_i

    real, intent(out) :: laplace_i

    integer :: j, neigh_idx
    real :: particle_laplace, deriv

    particle_laplace = 0.

    neighbour_loop: do j = 1, num_neighbours
        neigh_idx = indices(j)
        call cubic_kernel(distances(j), h, deriv)
        particle_laplace = particle_laplace + &
            ( (m_list(neigh_idx)/rho_list(neigh_idx))*(T_i - T_field(neigh_idx))*deriv)
    end do neighbour_loop

    laplace_i = 2.0*alpha_i*particle_laplace

end subroutine discrete_laplacian
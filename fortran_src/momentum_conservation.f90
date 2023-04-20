subroutine momentum_conservation(h, X_i, P_i, rho_i, nu, &
    num_domain_neighbours, domain_indices, domain_distances, &
    num_virtual_neighbours, virtual_indices, virtual_distances, &
    num_domain_particles, domain_masses, domain_positions, domain_densities, domain_pressures, &
    num_virtual_particles, virtual_masses, virtual_positions, virtual_densities, virtual_pressures, &
    dv_dt)
    !
    ! DOCUMENTATION PENDING !!!
    !
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_domain_neighbours, num_virtual_neighbours
    integer, intent(in) :: num_domain_particles, num_virtual_particles
    real, intent(in) :: nu
    real, intent(in) :: P_i
    real, intent(in) :: rho_i
    real, dimension(2), intent(in) :: X_i
    integer, dimension(num_domain_particles), intent(in) :: domain_indices
    real, dimension(num_domain_particles), intent(in) :: domain_distances
    real, dimension(num_domain_particles), intent(in) :: domain_masses
    real, dimension(num_domain_particles, 2), intent(in) :: domain_positions
    real, dimension(num_domain_particles), intent(in) :: domain_densities
    real, dimension(num_domain_particles), intent(in) :: domain_pressures

    integer, dimension(num_virtual_particles), intent(in) :: virtual_indices
    real, dimension(num_virtual_particles), intent(in) :: virtual_distances
    real, dimension(num_virtual_particles), intent(in) :: virtual_masses
    real, dimension(num_virtual_particles, 2), intent(in) :: virtual_positions
    real, dimension(num_virtual_particles), intent(in) :: virtual_densities
    real, dimension(num_virtual_particles), intent(in) :: virtual_pressures

    real, dimension(2), intent(out) :: dv_dt

    integer :: j, dom_neigh_id, virt_neigh_id
    real :: rd_ij, rv_ij, md_j, mv_j, Pd_j, Pv_j, rhod_j, rhov_j, W, dWdq

    real, dimension(2) :: Xd_j, Xv_j, grad_W
    real, dimension(2) :: pressure_term, viscous_term

    dv_dt = (/0., 0./)

    viscous_term = (/0., 0./)

    neighbour_loop: do j = 1, num_domain_neighbours
        dom_neigh_id = domain_indices(j)
        rd_ij = domain_distances(j)

        nonzero: if (rd_ij > 1e-5) then
            md_j = domain_masses(dom_neigh_id)
            Xd_j = domain_positions(dom_neigh_id, :)
            rhod_j = domain_densities(dom_neigh_id)

            call cubic_kernel(h, X_i, Xd_j, W, dWdq, grad_W)
            viscous_term = viscous_term + &
            2.0*nu*(md_j/rhod_j)*((X_i - Xd_j)/rd_ij)*grad_W
        end if nonzero

    end do neighbour_loop

    ! fs_i = self.compute_surface_forces(i, distances, indices, num_neighbours, h, dx, dy)

    call pressure_gradient(h, X_i, P_i, rho_i, &
        num_domain_neighbours, domain_indices, domain_distances, &
        num_virtual_neighbours, virtual_indices, virtual_distances, &
        num_domain_particles, domain_masses, domain_positions, domain_densities, domain_pressures, &
        num_virtual_particles, virtual_masses, virtual_positions, virtual_densities, virtual_pressures, &
        pressure_term)

    dv_dt = pressure_term + viscous_term
    !dv_dt = pressure_gradient + viscous_forces + fs_i

end subroutine momentum_conservation
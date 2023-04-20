subroutine pressure_gradient(h, X_i, P_i, rho_i, &
    num_domain_neighbours, domain_indices, domain_distances, &
    num_virtual_neighbours, virtual_indices, virtual_distances, &
    num_domain_particles, domain_masses, domain_positions, domain_densities, domain_pressures, &
    num_virtual_particles, virtual_masses, virtual_positions, virtual_densities, virtual_pressures, &
    grad_P)
    !
    ! DOCUMENTATION PENDING !!!
    !
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_domain_neighbours, num_virtual_neighbours
    integer, intent(in) :: num_domain_particles, num_virtual_particles
    real, dimension(2), intent(in) :: X_i
    real, intent(in) :: P_i
    real, intent(in) :: rho_i
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
    
    real, dimension(2), intent(out) :: grad_P

    integer :: j, dom_neigh_id, virt_neigh_id
    real :: rd_ij, rv_ij, md_j, mv_j, Pd_j, Pv_j, rhod_j, rhov_j, W, dWdq

    real, dimension(2) :: Xd_j, Xv_j, grad_W

    grad_P = (/0., 0./)

    neighbour_loop: do j = 1, num_domain_neighbours
        dom_neigh_id = domain_indices(j)
        rd_ij = domain_distances(j)

        nonzero: if (rd_ij > 1e-5) then
            md_j = domain_masses(dom_neigh_id)
            Xd_j = domain_positions(dom_neigh_id, :)
            Pd_j = domain_pressures(dom_neigh_id)
            rhod_j = domain_densities(dom_neigh_id)

            call cubic_kernel(h, X_i, Xd_j, W, dWdq, grad_W)

            grad_P = grad_P + md_j * ((P_i/rho_i**2) + (Pd_j/rhod_j**2))*grad_W
        end if nonzero
    end do neighbour_loop

    virtual_loop: do j = 1, num_virtual_neighbours
        virt_neigh_id = virtual_indices(j)
        rv_ij = virtual_distances(j)

        nonzero_2: if (rv_ij > 1e-5) then
            mv_j = virtual_masses(virt_neigh_id)
            Xv_j = virtual_positions(virt_neigh_id, :)
            Pv_j = virtual_pressures(virt_neigh_id)
            rhov_j = virtual_densities(virt_neigh_id)

            call cubic_kernel(h, X_i, Xv_j, W, dWdq, grad_W)
            grad_P = grad_P + mv_j * ((P_i/rho_i**2) + (Pv_j/rhov_j**2))*grad_W
        end if nonzero_2
    end do virtual_loop

end subroutine pressure_gradient
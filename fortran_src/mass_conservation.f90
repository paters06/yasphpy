subroutine mass_conservation(h, X_i, vel_i, &
    num_domain_neighbours, domain_indices, domain_distances, &
    num_virtual_neighbours, virtual_indices, virtual_distances, &
    num_domain_particles, domain_masses, domain_positions, domain_velocities, &
    num_virtual_particles, virtual_masses, virtual_positions, virtual_velocities, &
    drho_dt)
    !   Inputs:
    !   -------
    !   h: kernel function support radius
    !   X_i: position of particle at i
    !   vel_i: velocity of particle at i
    !   num_domain_neighbours: number of neighbour particles from the domain
    !   domain_indices: indices of neighbour particles from the domain
    !   domain_distances: distances from the neighbour domain particles to the i-particle
    !
    !   num_virtual_neighbours: number of neighbour particles from the virtual boundary
    !   virtual_indices: indices of neighbour particles from the virtual boundary
    !   virtual_distances: distances from the neighbour virtual particles to the i-particle
    !   
    !   domain_masses: masses of the domain particles
    !   domain_positions: positions of the domain particles
    !   domain_velocities: velocities of the domain particles
    !
    !   virtual_masses: masses of the virtual particles
    !   virtual_positions: positions of the virtual particles
    !   virtual_velocities: velocities of the virtual particles
    !
    !   Other variables:
    !   -------
    !   dom_neigh_id: index of a domain neighbour
    !   virt_neigh_id: index of a virtual neighbour
    !   rd_ij: distance from the domain neighbour to the i-particle
    !   rv_ij: distance from the virtual neighbour to the i-particle
    !   Xd_j: position of the domain neighbour
    !   Xv_j: position of the virtual neighbour
    !   md_j: mass of the domain neighbour
    !   mv_j: mass of the virtual neighbour
    !   veld_j: velocity of the domain neighbour
    !   velv_j: velocity of the virtual neighbour
    !
    use smoothing_functions
    implicit none

    real, intent(in) :: h
    integer, intent(in) :: num_domain_neighbours, num_virtual_neighbours
    integer, intent(in) :: num_domain_particles, num_virtual_particles
    real, dimension(2), intent(in) :: X_i
    real, dimension(2), intent(in) :: vel_i
    integer, dimension(num_domain_particles), intent(in) :: domain_indices
    real, dimension(num_domain_particles), intent(in) :: domain_distances
    real, dimension(num_domain_particles), intent(in) :: domain_masses
    real, dimension(num_domain_particles, 2), intent(in) :: domain_positions
    real, dimension(num_domain_particles, 2), intent(in) :: domain_velocities

    integer, dimension(num_virtual_particles), intent(in) :: virtual_indices
    real, dimension(num_virtual_particles), intent(in) :: virtual_distances
    real, dimension(num_virtual_particles), intent(in) :: virtual_masses
    real, dimension(num_virtual_particles, 2), intent(in) :: virtual_positions
    real, dimension(num_virtual_particles, 2), intent(in) :: virtual_velocities

    real, intent(out) :: drho_dt

    integer :: j, dom_neigh_id, virt_neigh_id
    real :: rd_ij, rv_ij, md_j, mv_j, W, dWdq

    real, dimension(2) :: Xd_j, Xv_j, veld_j, velv_j, grad_W

    drho_dt = 0.0

    domain_loop: do j = 1, num_domain_neighbours
        dom_neigh_id = domain_indices(j)
        rd_ij = domain_distances(j)
        Xd_j = domain_positions(dom_neigh_id, :)
        md_j = domain_masses(dom_neigh_id)
        veld_j = domain_velocities(dom_neigh_id, :)

        nonzero_1: if (rd_ij > 1e-5) then
            call cubic_kernel(h, X_i, Xd_j, W, dWdq, grad_W)
            drho_dt = drho_dt + md_j*dot_product((vel_i - veld_j), grad_W)
        end if nonzero_1

    end do domain_loop

    ! write (*,*) drho_dt

    virtual_loop: do j = 1, num_virtual_neighbours
        virt_neigh_id = virtual_indices(j)
        rv_ij = virtual_distances(j)
        Xv_j = virtual_positions(virt_neigh_id, :)
        mv_j = virtual_masses(virt_neigh_id)
        velv_j = virtual_velocities(virt_neigh_id, :)

        nonzero_2: if (rv_ij > 1e-5) then
            call cubic_kernel(h, X_i, Xv_j, W, dWdq, grad_W)
            drho_dt = drho_dt + mv_j*dot_product((vel_i - velv_j), grad_W)
        end if nonzero_2
    end do virtual_loop

    ! write (*,*) drho_dt

end subroutine mass_conservation
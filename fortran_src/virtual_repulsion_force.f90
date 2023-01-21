subroutine virtual_repulsion_force(cutoff_radius, X_i, v_max, &
                                   num_virtual_particles, &
                                   num_virtual_neighbours, virtual_indices, &
                                   virtual_particle_list, F_iv)
    !   i-particle: given fluid particle
    !   v-particle: given virtual particle
    !   ro: cut-off distance
    !   X_iv: difference on the position vectors for i-particle and v-particle
    !   r_iv: distance from the center of mass between the i-particle and the v-particle
    !   distance_ratio: r_o/r_iv
    !   v_max: speed magnitude
    implicit none

    integer, intent(in) :: num_virtual_neighbours
    integer, intent(in) :: num_virtual_particles
    real, intent(in) :: cutoff_radius
    real, intent(in) :: v_max
    real, dimension(2), intent(in) :: X_i
    integer, dimension(num_virtual_particles), intent(in) :: virtual_indices
    real, dimension(num_virtual_particles, 2), intent(in) :: virtual_particle_list
    
    real, dimension(2), intent(out) :: F_iv

    integer :: j, virtual_neigh_idx
    real :: n1, n2, psi, distance_ratio, r_iv
    real, dimension(2) :: X_v, X_iv

    n1 = 12.
    n2 = 4.

    psi = v_max**2.

    F_iv = (/0., 0./)
    neighbour_loop: do j = 1, num_virtual_neighbours
        virtual_neigh_idx = virtual_indices(j)
        X_v = virtual_particle_list(virtual_neigh_idx, :)
        X_iv = X_i - X_v
        r_iv = norm2(X_iv)
        distance_ratio = cutoff_radius/r_iv
        nonzero: if (distance_ratio <= 1.0) then
            F_iv = F_iv + psi*(distance_ratio**n1 - distance_ratio**n2)*(X_iv/r_iv**2)
        end if nonzero
    end do neighbour_loop
end subroutine virtual_repulsion_force
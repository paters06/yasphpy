subroutine find_neighbour(arr, m, n, vec, p, radius, distances, indices, num_neighbours)
    implicit none

    integer, intent(in) :: m
    integer, intent(in) :: n
    integer, intent(in) :: p
    integer :: i
    integer, intent(out) :: num_neighbours

    real, dimension(m, n) :: arr
    real, dimension(p) :: vec
    real, intent(in) :: radius
    real :: dist

    real, dimension(m), intent(out) :: distances
    integer, dimension(m), intent(out) :: indices

    num_neighbours = 0

    outer: do i = 1, m
            dist = sqrt((arr(i, 1) - vec(1))**2 + (arr(i, 2) - vec(2))**2)
            ! write(*,*)'Dist = ', dist
            if (dist <= radius) then
                num_neighbours = num_neighbours + 1
                distances(num_neighbours) = dist
                indices(num_neighbours) = i
            end if
    end do outer

end subroutine find_neighbour
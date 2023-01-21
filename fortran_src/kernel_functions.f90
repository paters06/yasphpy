module smoothing_functions
    implicit none
    contains
    subroutine cubic_kernel(h, xi, xj, W, dWdq, grad)
        implicit none

        real, intent(in) :: h
        real, intent(out) :: W
        real, intent(out) :: dWdq
        real, dimension(2), intent(in) :: xi
        real, dimension(2), intent(in) :: xj
        real, dimension(2), intent(out) :: grad

        real :: delta_x, delta_y, r_ij, q, alpha_D
        real, parameter :: PI = 3.1416

        q = 0.
        alpha_D = 0.
        delta_x = 0.
        delta_y = 0.

        delta_x = xj(1) - xi(1)
        delta_y = xj(2) - xi(2)
        
        r_ij = sqrt(delta_x**2 + delta_y**2)

        alpha_D = 15/(7*PI*h**2)

        non_zero: if (abs(r_ij) > 1e-5) then
            q = r_ij/h

            spline: if (q >= 0.0 .and. q <= 1.0) then
                W = alpha_D*((2./3) - q*q + 0.5*q**3)
            else if (q > 1.0 .and. q <= 2.0) then
                W = alpha_D*((1./6)*(2.0 - q)**3)
            else
                W = 0.0
            end if spline

            derivative: if (q >= 0.0 .and. q <= 1.0) then
                dWdq = alpha_D*(-2.0*q + 1.5*q**2)
            else if (q > 1.0 .and. q <= 2.0) then
                dWdq = alpha_D*0.5*(-(2.0 - q)**2)
            else
                dWdq = 0.0
            end if derivative
            
            grad(1) = dWdq*((1.0/h)*(delta_x/r_ij))
            grad(2) = dWdq*((1.0/h)*(delta_y/r_ij))
        else
            W = 0.
            dWdq = 0.0
            grad(1) = 0.0
            grad(2) = 0.0
        end if non_zero
    end subroutine cubic_kernel
end module smoothing_functions
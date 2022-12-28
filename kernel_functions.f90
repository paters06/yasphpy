module smoothing_functions
    implicit none
    contains
    subroutine cubic_kernel(r_ij, h, dx, dy, W, dWdq, grad)
        implicit none

        real, intent(in) :: r_ij
        real, intent(in) :: h
        real, intent(in) :: dx
        real, intent(in) :: dy
        real, intent(out) :: W
        real, intent(out) :: dWdq
        real, dimension(2), intent(out) :: grad

        real :: q, alpha_D
        real, parameter :: PI = 3.1416

        q = 0.
        alpha_D = 0.
        W = 0.

        alpha_D = 15/(7*PI*h**2)

        spline: if (q >= 0.0 .and. q <= 1.0) then
            W = alpha_D*((2./3) - q*q + 0.5*q**3)
        else if (q > 1.0 .and. q <= 2.0) then
            W = alpha_D*((1./6)*(2.0 - q)**3)
        else
            W = 0.0
        end if spline

        non_zero: if (abs(r_ij) > 1e-5) then
            q = r_ij/h

            derivative: if (q >= 0.0 .and. q <= 1.0) then
                dWdq = alpha_D*(-2.0*q + 1.5*q**2)
            else if (q > 1.0 .and. q <= 2.0) then
                dWdq = alpha_D*0.5*(-(2.0 - q)**2)
            else
                dWdq = 0.0
            end if derivative
            
            grad(1) = dWdq*((1.0/h)*(dx/r_ij))
            grad(2) = dWdq*((1.0/h)*(dy/r_ij))
        else
            dWdq = 0.0
            grad(1) = 0.0
            grad(2) = 0.0
        end if non_zero
    end subroutine cubic_kernel
end module smoothing_functions
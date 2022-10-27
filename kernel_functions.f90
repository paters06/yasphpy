module kernel_functions
    implicit none
    contains
    subroutine cubic_kernel_derivative(r_ij, h, deriv)
        implicit none

        real, intent(in) :: r_ij
        real, intent(in) :: h
        real, intent(out) :: deriv

        real :: q, alpha_D
        real, parameter :: PI = 3.1416

        q = 0.
        alpha_D = 0.

        non_zero: if (abs(r_ij) > 1e-5) then
            q = r_ij/h

            alpha_D = 15/(7*PI*h**2)

            derivative: if (q >= 0.0 .and. q <= 1.0) then
                deriv = alpha_D*(-2*q + 1.5*q**2)*(1.0/(q*h))
            else if (q > 1.0 .and. q <= 2.0) then
                deriv = alpha_D*(1.0/2)*(-(2 - q)**2)*(1.0/(q*h))
            else
                deriv = 0.0
            end if derivative
        else
            deriv = 0.0
        end if non_zero
    end subroutine cubic_kernel_derivative
end module kernel_functions
import math
import numpy as np

# class KernelFunction:


def cubic_spline_kernel(x: np.ndarray, x_prime: np.ndarray, h: float) -> float:
    """
    Compute the kernel function through the cubic spline formula

    Input:
        - x (`np.ndarray'): position of the center of the kernel function

        - x_prime (`np.ndarray'): position of a given point
                             inside the kernel function

        - h (`float'): kernel function radius
    """
    q = np.linalg.norm(x - x_prime)/h
    # print(q)
    alpha_D = 15/(7*math.pi*h**2)

    if q >= 0.0 and q <= 1.0:
        return alpha_D*((2/3) - q**2 + 0.5*q**3)
    elif q > 1.0 and q <= 2.0:
        return alpha_D*((1/6)*(2 - q)**3)
    else:
        return 0


def cubic_spline_kernel_derivative(x: np.ndarray, x_prime: np.ndarray,
                                   h: float):
    q = np.linalg.norm(x - x_prime)/h
    
    alpha_D = 15/(7*math.pi*h**2)
    # print("q: {:.3f}".format(q))
    # print("h: {:.3f}".format(h))
    # print("a_D: {:.3f}".format(alpha_D))

    if q >= 0.0 and q <= 1.0:
        # print("a_D: {:.3f}".format(alpha_D))
        # print("x:", x)
        # print("x':", x_prime)
        # print(np.linalg.norm(x-x_prime))
        # print(np.sqrt((x[0] - x_prime[0])**2 + (x[1] - x_prime[1])**2))
        # print("q: {:.3f}".format(q))
        # print("Polynomial: {:.3f}".format(-2*q + 1.5*q**2))
        return alpha_D*(-2*q + 1.5*q**2)
    elif q > 1.0 and q <= 2.0:
        # print("a_D: {:.3f}".format(alpha_D))
        # print("x:", x)
        # print("x':", x_prime)
        # print(np.linalg.norm(x-x_prime))
        # print(np.sqrt((x[0] - x_prime[0])**2 + (x[1] - x_prime[1])**2))
        # print("q: {:.3f}".format(q))
        # print("Polynomial: {:.3f}".format((-(2 - q)**2)/2))
        return alpha_D*(-(2 - q)**2)/2
    else:
        return 0


def lucy_quartic_kernel(x: np.ndarray, x_prime: np.ndarray, h: float) -> float:
    """
    Compute the kernel function through the cubic spline formula

    Input:
        - x (`np.ndarray'): position of the center of the kernel function

        - x_prime (`np.ndarray'): position of a given point
                             inside the kernel function

        - h (`float'): kernel function radius
    """
    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 5/(math.pi*h**2)

    if q >= 0.0 and q <= 1.0:
        return alpha_D*((1 + 3*q)*(1 - q**3))
    else:
        return 0


def lucy_quartic_kernel_derivative(x: np.ndarray, x_prime: np.ndarray,
                                   h: float) -> float:
    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 5/(math.pi*h**2)
    if q >= 0.0 and q <= 1.0:
        return alpha_D*(-12*q + 24*q**2 - 12*q**3)
    else:
        return 0


def new_quartic_kernel(x: np.ndarray, x_prime: np.ndarray, h: float) -> float:
    """
    Compute the kernel function through the cubic spline formula

    Input:
        - x (`np.ndarray'): position of the center of the kernel function

        - x_prime (`np.ndarray'): position of a given point
                             inside the kernel function

        - h (`float'): kernel function radius
    """
    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 15/(7*math.pi*h**2)

    if q >= 0.0 and q <= 2.0:
        return alpha_D*((2/3) - (9/8)*q**2 + (19/24)*q**3 - (5/32)*q**4)
    else:
        return 0


def new_quartic_kernel_derivative(x: np.ndarray, x_prime: np.ndarray,
                                  h: float) -> float:
    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 15/(7*math.pi*h**2)

    if q >= 0.0 and q <= 2.0:
        return alpha_D*(- (18/8)*q + (57/24)*q**2 - (20/32)*q**3)
    else:
        return 0


def quintic_kernel(x: np.ndarray, x_prime: np.ndarray, h: float) -> float:
    """
    Compute the kernel function through the cubic spline formula

    Input:
        - x (`np.ndarray'): position of the center of the kernel function

        - x_prime (`np.ndarray'): position of a given point
                             inside the kernel function

        - h (`float'): kernel function radius
    """
    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 7/(478*math.pi*h**2)

    if q >= 0.0 and q <= 1.0:
        return alpha_D*((3 - q)**5 - 6*(2 - q)**5 + 15*(1 - q)**5)
    elif q > 1.0 and q <= 2.0:
        return alpha_D*((3 - q)**5 - 6*(2 - q)**5)
    elif q > 2.0 and q <= 3.0:
        return alpha_D*((3 - q)**5)
    else:
        return 0


def quintic_kernel_derivative(x: np.ndarray, x_prime: np.ndarray,
                              h: float) -> float:

    q = np.linalg.norm(x - x_prime)/h
    alpha_D = 7/(478*math.pi*h**2)

    if q >= 0.0 and q <= 1.0:
        return alpha_D*(-120*q + 120*q**3 - 50*q**4)
    elif q > 1.0 and q <= 2.0:
        return alpha_D*(75 - 420*q + 450*q**2 - 180*q**3 + 25*q**4)
    elif q > 2.0 and q <= 3.0:
        return alpha_D*(405 + 540*q - 270*q**2 + 60*q**3 - 5*q**4)
    else:
        return 0

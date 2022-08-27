import math
import numpy as np


def cubic_spline(x: np.ndarray, x_prime: np.ndarray, h: float) -> float:
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

    if q >= 0.0 and q <= 1.0:
        return alpha_D*((2/3) - q**2 + 0.5*q**3)
    elif q > 1.0 and q <= 2.0:
        return alpha_D*((1/6)*(2 - q**3))
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

from numpy import f2py

# with open("cubic_kernel.f90") as sourcefile:
#     sourcecode = sourcefile.read()

# f2py.compile(sourcecode, modulename='kernel_function',
#              extension='.f90', verbose=False)


# import kernel_function

# print(kernel_function.cubic_kernel_derivative.__doc__)

# import pressure_gradient
import mass_conservation
import momentum_conservation
import virtual_repulsion_force

# print(pressure_gradient.pressure_gradient.__doc__)
# print(mass_conservation.mass_conservation.__doc__)
print(momentum_conservation.momentum_conservation.__doc__)
# print(virtual_repulsion_force.virtual_repulsion_force.__doc__)

# import ball

# ball.compute_maximum()

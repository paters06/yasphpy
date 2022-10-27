from numpy import f2py

with open("cubic_kernel.f90") as sourcefile:
    sourcecode = sourcefile.read()

f2py.compile(sourcecode, modulename='kernel_function',
             extension='.f90', verbose=False)


import kernel_function

print(kernel_function.cubic_kernel_derivative.__doc__)

# import ball

# ball.compute_maximum()
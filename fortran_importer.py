from numpy import f2py

with open("find_neighbour.f90") as sourcefile:
    sourcecode = sourcefile.read()

f2py.compile(sourcecode, modulename='neighbour',
             extension='.f90', verbose=False)


import neighbour

print(neighbour.find_neighbour.__doc__)

# import ball

# ball.compute_maximum()
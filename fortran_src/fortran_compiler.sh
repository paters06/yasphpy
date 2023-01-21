#!/bin/bash
#python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 discrete_laplacian.f90 -m discrete_laplacian -h discrete_laplacian.pyf
#python3 -m numpy.f2py -c --fcompiler=gnu95 discrete_laplacian.pyf kernel_functions.f90 discrete_laplacian.f90 
# python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 -m kernel_functions -h kernel_functions.pyf
# python3 -m numpy.f2py -c --fcompiler=gnu95 kernel_functions.pyf kernel_functions.f90

# Building and linking virtual_repulsion_force
python3 -m numpy.f2py --overwrite-signature --quiet virtual_repulsion_force.f90 -m virtual_repulsion_force -h virtual_repulsion_force.pyf
python3 -m numpy.f2py -c --fcompiler=gnu95 virtual_repulsion_force.pyf virtual_repulsion_force.f90

# Building and linking pressure_gradient with kernel_functions
python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 pressure_gradient.f90 -m pressure_gradient -h pressure_gradient.pyf
python3 -m numpy.f2py -c --fcompiler=gnu95 pressure_gradient.pyf kernel_functions.f90 pressure_gradient.f90

# Building and linking mass_conservation with kernel_functions
python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 mass_conservation.f90 -m mass_conservation -h mass_conservation.pyf
python3 -m numpy.f2py -c --fcompiler=gnu95 mass_conservation.pyf kernel_functions.f90 mass_conservation.f90

# Building and linking momentum_conservation with kernel_functions
python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 momentum_conservation.f90 -m momentum_conservation -h momentum_conservation.pyf
python3 -m numpy.f2py -c --fcompiler=gnu95 momentum_conservation.pyf kernel_functions.f90 momentum_conservation.f90

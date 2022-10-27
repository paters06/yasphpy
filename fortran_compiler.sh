#!/bin/bash
python3 -m numpy.f2py --overwrite-signature --quiet kernel_functions.f90 discrete_laplacian.f90 -m discrete_laplacian -h discrete_laplacian.pyf
python3 -m numpy.f2py -c --fcompiler=gnu95 discrete_laplacian.pyf kernel_functions.f90 discrete_laplacian.f90 
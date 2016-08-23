#!/bin/bash
mkdir build
cd build/
cmake ..
make
mpirun -np 4 application-MPI
mpirun -np 5 application-MPI
cd ..

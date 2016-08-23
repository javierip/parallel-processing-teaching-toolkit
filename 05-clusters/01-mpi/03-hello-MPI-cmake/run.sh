#!/bin/bash
mkdir build
cd build/
cmake ..
make
mpirun -np 8 application-MPI
cd ..

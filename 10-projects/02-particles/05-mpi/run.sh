#!/bin/bash
mkdir build
cd build/
cmake ..
make
mpirun -np 8 application -n 100 -o output-mpi-100-particles.txt
cd ..

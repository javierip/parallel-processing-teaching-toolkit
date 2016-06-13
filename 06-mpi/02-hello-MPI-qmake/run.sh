#!/bin/bash
qmake 
make
mpirun -np 8 application-MPI

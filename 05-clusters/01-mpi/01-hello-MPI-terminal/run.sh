#!/bin/bash
mpicc -o application-MPI main.c
mpirun -np 8 application-MPI

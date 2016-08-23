mpicc -o application-MPI main.c
mpirun -hostfile my_hostfile -np 8 application-MPI 


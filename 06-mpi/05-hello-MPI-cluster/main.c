#include <mpi.h> 
#include <stdio.h>

int main(int argc, char *argv[]) {
    int myrank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from process %d of %d\n", myrank, size);
    MPI_Finalize();

    return 0;
} 

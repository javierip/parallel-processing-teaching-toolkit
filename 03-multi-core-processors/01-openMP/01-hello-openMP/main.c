#include <stdio.h>
#include <omp.h>

int main () 
{
    int numberOfThreads, idThread;
    /* Multiple threads running in parallel from here */
    #pragma omp parallel private(idThread)
    {
        /* Obtenemos el id de cada thread */
        idThread = omp_get_thread_num();
        printf("Hello! my id is %d\n", idThread);

        if (idThread == 0)
        {
            numberOfThreads = omp_get_num_threads();
            printf("Total number of threads = %d\n", numberOfThreads);
        }
    }

    /* All threads join into one */
    printf("One thread here\n");

    return 0;
} 

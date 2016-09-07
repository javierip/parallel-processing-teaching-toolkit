#include <stdio.h>
#include <omp.h>

int main ()  {
    int numberOfThreads, threadID;
    printf("Setting a fixed number of threads. In this case 8\n");
    omp_set_num_threads(8);

    numberOfThreads = omp_get_num_threads();
    printf("The total number of threads is %d\n", numberOfThreads);

    #pragma omp parallel private(threadID)
    {
        // each thread know its ID using threadID as private
        threadID = omp_get_thread_num();
        printf("Hello! my ID is %d\n", threadID);
        if (threadID == 0) {
            numberOfThreads = omp_get_num_threads();
            printf("I am the thread 0 and the total numer is %d\n", numberOfThreads);
        }
    }

    printf("Now we use 5 threads\n");

    omp_set_num_threads(5);

    numberOfThreads = omp_get_num_threads();
    printf("Total number of threads %d\n", numberOfThreads);
    #pragma omp parallel
    {
        threadID = omp_get_thread_num();
        printf("Hello! my ID is %d\n", threadID);
        if (threadID == 0) {
            numberOfThreads = omp_get_num_threads();
            printf("I am the thread 0 and the total numer is %d\n", numberOfThreads);
        }
    }

    return 0;
}

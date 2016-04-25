#include <stdio.h>
#include <omp.h>

int main () {
    int numberOfThreads, threadID;

    #pragma omp parallel private(threadID)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                threadID = omp_get_thread_num();
                numberOfThreads = omp_get_num_threads();
                printf("Thread %d of %d calculates section 1\n", threadID, numberOfThreads);
            }
            #pragma omp section
            {
                threadID = omp_get_thread_num();
                numberOfThreads = omp_get_num_threads();
                printf("Thread %d of %d calculates section 2\n", threadID, numberOfThreads);
            }
            #pragma omp section
            {
                threadID = omp_get_thread_num();
                numberOfThreads = omp_get_num_threads();
                printf("Thread %d of %d calculates section 3\n", threadID, numberOfThreads);
            }
            #pragma omp section
            {
                threadID = omp_get_thread_num();
                numberOfThreads = omp_get_num_threads();
                printf("Thread %d of %d calculates section 4\n", threadID, numberOfThreads);
            }
        }
    }

    return 0;
}

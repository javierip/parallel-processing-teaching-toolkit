#include <stdio.h>
#include <omp.h>
#define CHUNKSIZE 2
#define N    10

int main () {
    int i, chunk;
    int numberOfThreads, threadID;
    int a[N], b[N], c[N];

    // intial values
    for (i = 0; i < N; i++)
        a[i] = b[i] = i * 1.0;

    chunk = CHUNKSIZE;

    printf("Static scheduling\n");
    #pragma omp parallel shared(a, b, c, chunk) private(i, threadID)
    {
        #pragma omp for schedule(static, chunk)
        for (i = 0; i < N; i++) {
            threadID = omp_get_thread_num();
            numberOfThreads = omp_get_num_threads();
            c[i] = a[i] + b[i];
            printf("Thread %d of %d is calculating the iteration %d\n", threadID, numberOfThreads, i);
        }
    }

    printf("Dynamic scheduling\n");
    #pragma omp parallel shared(a, b, c, chunk) private(i, threadID)
    {
        #pragma omp for schedule(dynamic, chunk)
        for (i = 0; i < N; i++) {
            threadID = omp_get_thread_num();
            numberOfThreads = omp_get_num_threads();
            c[i] = a[i] + b[i];
            printf("Thread %d of %d is calculating the iteration %d\n", threadID, numberOfThreads, i);
        }
    }
}

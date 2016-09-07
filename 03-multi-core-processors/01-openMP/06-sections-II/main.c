#include <omp.h>

#define MAX_NUMBER_OF_ELEMENTS 10

int main ()
{
    int i;
    int threadID, totalNumberThreads;
    float vectorA[MAX_NUMBER_OF_ELEMENTS], vectorB[MAX_NUMBER_OF_ELEMENTS],
            vectorC[MAX_NUMBER_OF_ELEMENTS], vectorD[MAX_NUMBER_OF_ELEMENTS];

    // data initialization
    for (i=0; i < MAX_NUMBER_OF_ELEMENTS; i++)
    {
        vectorA[i] = i * 1.5;
        vectorB[i] = i + 22.35;
    }

    #pragma omp parallel shared(vectorA,vectorB,vectorC,vectorD)\
    private(i, threadID, totalNumberThreads)
    {
        threadID = omp_get_thread_num();
        totalNumberThreads = omp_get_num_threads();

        #pragma omp sections nowait
        {
            #pragma omp section
            for (i=0; i < MAX_NUMBER_OF_ELEMENTS; i++)
            {
                vectorC[i] = vectorA[i] + vectorB[i];
                printf("Thread %d of %d calculates i = %d (section 1)\n", threadID, totalNumberThreads, i);
            }
            #pragma omp section
            for (i=0; i < MAX_NUMBER_OF_ELEMENTS; i++)
            {
                vectorD[i] = vectorA[i] * vectorB[i];
                printf("Thread %d of %d calculates i = %d (section 2)\n", threadID, totalNumberThreads, i);
            }
        }  /* end of sections */
    }  /* end of parallel section */

    return 0;
}

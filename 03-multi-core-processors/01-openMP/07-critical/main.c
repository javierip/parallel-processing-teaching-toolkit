#include <stdio.h>
#include <omp.h>

int main() {
    int addition, threadID;
    addition = 0;
    
    printf("Using critical\n");
    #pragma omp parallel shared(addition) private(threadID)
    {
        threadID = omp_get_thread_num();
        #pragma omp critical
        {
            addition = addition + 1;
            printf("Thread %d is accessing value %d\n", threadID, addition);
        }
    }
    printf("Final value of the addition is %d\n", addition);

    addition = 0;
    printf("Not using critical\n");
    #pragma omp parallel shared(addition) private(threadID)
    {
        threadID = omp_get_thread_num();
        //#pragma omp critical
        {
            addition = addition + 1;
            printf("Thread %d is accessing value %d\n", threadID, addition);
        }
    }
    printf("Final value of the addition is %d\n", addition);
}

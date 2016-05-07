#include <stdio.h>
#include <omp.h>

#define ROW_MAX_VALUE 10
#define COLUMN_MAX_VALUE 10

int main () {
    int i, j;
    int totalNumberThreads, threadID;
    int actualRow, actualColumn, sum;
    int a[ROW_MAX_VALUE], c[ROW_MAX_VALUE],
            b[COLUMN_MAX_VALUE][ROW_MAX_VALUE];

    actualColumn = COLUMN_MAX_VALUE;
    actualRow = ROW_MAX_VALUE;
    for (i = 0; i < actualRow; i++)
        c[i] = i;
    for (i = 0; i < actualColumn; i++)
        for (j = 0; j < actualRow; j++)
            b[i][j] = i + j;

    #pragma omp parallel for default(none) \
    private(threadID, i, j, sum) shared(actualColumn, actualRow, a, b, c, totalNumberThreads)
    for (i = 0; i < actualColumn; i++)
    {
        threadID = omp_get_thread_num();
        totalNumberThreads = omp_get_num_threads();
        sum = 0;
        for (j = 0; j < actualRow; j++)
            sum += b[i][j] * c[j];
        a[i] = sum;
        printf("Thread %d of %d calculates i = %d\n", threadID, totalNumberThreads, i);
    }

    printf("The content of vector A is:\n");
    for(i = 0; i < COLUMN_MAX_VALUE; i++)
    {
        printf(" %d ", a[i]);
    }
    printf("\n");

    return 0;
}

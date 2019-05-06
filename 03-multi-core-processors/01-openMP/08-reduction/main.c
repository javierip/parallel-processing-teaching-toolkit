#include <stdio.h>
#include <omp.h>

#define N 100

void populate_vector(int vector[], int value, int length)
{
    for(int i=0; i<length; i++)
        vector[i] = value;
}

void reduction_example_1()
{
    int vector [N] = {0};
    int sum=0;

    populate_vector(vector, 1, N);

#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<N; ++i)
        sum += vector[i];

    printf("The reducion in example 1 is %d\n", sum);
}

void reduction_example_2()
{
    int a[N], b[N], result,i, chunk;

    chunk = 10;
    result = 0.0;

    populate_vector(a, 1, N);
    populate_vector(b, 3, N);

#pragma omp parallel for \
    default(shared) private(i) \
    schedule(static,chunk) \
    reduction(+:result)

    for (i=0; i < N; i++)
        result = result + (a[i] * b[i]);

    printf("The reducion in example 2 is %d\n",result);
}

int main()
{
    reduction_example_1();
    reduction_example_2();
    return 0;
}

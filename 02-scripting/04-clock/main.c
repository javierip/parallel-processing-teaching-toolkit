#include <time.h>
#include <stdio.h>

void my_expensive_function()
{
  int i;
  double number = 0;
  for (i = 0; i < 10000000; i++)
  {
    number = number + (double)i / 50.0f;
  }
}

int main()
{
  clock_t tic = clock();

  my_expensive_function();

  clock_t toc = clock();

  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  return 0;
}
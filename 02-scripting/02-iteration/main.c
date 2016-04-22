#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("%s, there are %d arguments\n", "Hello there !", argc);
    int i;

    for(i = 0; i < argc; i++)
    {
        printf("Argument %d is %s\n", i, argv[i]);
    }

    return 0;
}

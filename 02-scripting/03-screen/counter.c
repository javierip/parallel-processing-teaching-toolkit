#include <stdio.h>
#include <unistd.h>

int main() {
    int i;
    for(i = 120; i >= 0; i-- )
    {
        printf("Second %d \n", i);
        sleep(1);
   }
    printf("Time elapsed\n");
    printf("Press ENTER  Key to Continue\n");  
    getchar();
    return 0;
}

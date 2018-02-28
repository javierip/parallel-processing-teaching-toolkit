#include "lib.h"



void get_CUDAinfo (){
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}
bool free_memGPU (float *arr1,float *arr2,float *arr3){

    // Free device global memory
    cudaError_t err;
    err = cudaFree(arr1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaFree(arr2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    err = cudaFree(arr3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }
    printf("Resources free from CUDA Device\n");
    return 1;

}

bool  init_vectors_CPU (float *arr_A,float *arr_B,int elements){



    // Verify that allocations succeeded
    if (arr_A == NULL || arr_B == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return 0;
    }

    // Initialize the host input vectors
    for (int i = 0; i < elements; ++i)
    {
        arr_A[i] = rand()/(float)RAND_MAX;
        arr_B[i] = rand()/(float)RAND_MAX;
    }
    return 1;
}
bool pedir_memoriaGPU(float *arr_A,float *arr_B,float *arr_C,size_t d_size){

    // Allocate the device input vector A
    cudaError_t err;
    err = cudaMalloc((void **) &arr_A, d_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    // Allocate the device input vector B

    err = cudaMalloc((void **)&arr_B, d_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    // Allocate the device output vector C

    err = cudaMalloc((void **)&arr_C, d_size);



    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

bool add_cpu(float *arr_A,float *arr_B,float *arr_C,int elements){

    for (int i = 0; i < elements; ++i)
    {
        arr_C[i]= arr_A[i] + arr_B[i];

    }
    return 1;

}

bool check_addition(float *arr_A,float *arr_B,float *arr_C,int elements){

    for (int i = 0; i < elements; ++i)
    {
        if (fabs(arr_A[i] + arr_B[i] - arr_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            return 0;
        }
    }
    return 1;

}

bool check (cudaError_t error ){
    if (error != cudaSuccess) return 0;
    //printf ("Error checkeado\n");
    return 1;
}


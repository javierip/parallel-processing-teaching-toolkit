
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <time.h>

/**
 * CUDA Kernel Device code
  * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 **/
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

     if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/// Functions Propotypes
//Get CUDA Platform Info
void get_CUDAinfo ();
//Free GPU Memory
bool free_memGPU (float *arr1,float *arr2,float *arr3);
//Load the CPU vectors
bool  init_vectors_CPU (float *arr_A,float *arr_B,int elements);
//Operation with CPU
bool add_cpu(float *arr_A,float *arr_B,float *arr_C,int elements);
//Check addition CPU vs GPU
bool check_addition(float *arr_A,float *arr_B,float *arr_C,int elements);
//Check CUDA Errors
bool check (cudaError_t error );

/// Host main routine 
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    get_CUDAinfo();
    clock_t start, end;
    double time_cpu,time_gpu;

    // Print the vector length to be used, and compute its size
    int numElements = 500;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);


    //Vectors on RAM 
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if(!init_vectors_CPU(h_A,h_B,numElements))printf( "Failed to init  vectors!\n");

    //Vectors on GPU Memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
 

     if (!check( cudaMalloc((void **)&d_A, size)))
     {
         printf( "Failed to allocate device vector A (error code %s)!\n");
         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
         return 0;
     }


     if (!check(cudaMalloc((void **)&d_B, size)))
     {
         printf( "Failed to allocate device vector B (error code %s)!\n");
         fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
         return 0;
     }




     if (!check(cudaMalloc((void **)&d_C, size)))
     {
         printf("Failed to allocate device vector C (error code %s)!\n");
         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
         return 0;
     }



    printf("Copy input data from the host memory to the CUDA device\n");
      if (!check(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)))
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }


    if (!check(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice)))
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 15;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    start = clock();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    end = clock();
    time_gpu= (double ) (end - start) / CLOCKS_PER_SEC * 1000;
  


    if (!check(cudaGetLastError()))
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");


    if (!check(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost)))
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        return 0;
    }

    
    start = clock();
    add_cpu(h_A,h_B,h_C,numElements);
    end = clock();
    time_cpu= (double ) (end - start) / CLOCKS_PER_SEC * 1000;
 

    // Verify that the result vector is correct  
    if(check_addition(h_A,h_B,h_C,numElements)) printf("Test PASSED\n");

  printf("Time GPU: %lf\n", time_gpu);
       printf("Time CPU: %lf\n", time_cpu);

    if(!free_memGPU(d_A,d_B,d_C))return 0;

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

/// Functions 
 
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


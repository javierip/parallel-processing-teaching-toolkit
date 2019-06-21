#include "lib.h"


/**
 * CUDA Kernel Device code
  * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 **/
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorReduce(const float *global_input_data, float *global_output_data, const int numElements)
{
    __shared__ float sdata[10];
    __shared__  int sindice[10];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x ) + threadIdx.x;
    sdata[tid] = global_input_data[i];
    sindice[tid] = tid;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (tid < s ) {
            if (sdata[tid] > sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
                sindice[tid] = sindice[tid + s];
            }
            __syncthreads();
        }
    }

    __syncthreads();

    if (tid == 0) {
        global_output_data[0] = sdata[0];

    }

    if (tid == 1) {
        global_output_data[1] = sindice[0];

    }

}




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



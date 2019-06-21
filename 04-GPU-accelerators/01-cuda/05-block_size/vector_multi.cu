/**  
  This example is based on the article titled "CUDA Pro Tip: Occupancy API Simplifies Launch Configuration".
  More info on https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
*/

#include "stdio.h"

__global__ void VectorMultiplicationKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount)
    {
        array[idx] *= array[idx];
    }
}

void launchMaxOccupancyKernel(int *array, int arrayCount)
{
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, VectorMultiplicationKernel, 0, 0);
    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;
    printf("Grid size is %d, array count is %d, min grid size is %d\n", gridSize, arrayCount, minGridSize);

    VectorMultiplicationKernel<<< gridSize, blockSize >>>(array, arrayCount);

    cudaDeviceSynchronize();

    // calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks,
                                                   VectorMultiplicationKernel,
                                                   blockSize, 0);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    float occupancy = (float) (maxActiveBlocks * blockSize / props.warpSize) /
            (float)(props.maxThreadsPerMultiProcessor / props.warpSize);

    printf("Device maxThreadsPerMultiProcessor %d\n", props.maxThreadsPerMultiProcessor);
    printf("Device warpSize %d\n", props.warpSize);

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n", blockSize, occupancy);
}

void initializeData(int *array, int count){
    for (int i = 0; i < count; i += 1) {
        array[i] = i;
    }
}

void resetData(int *array, int count){
    for (int i = 0; i < count; i += 1) {
        array[i] = 0;
    }
}

void verifyData(int *array, int count){
    bool isDataCorrect = true;

    for (int i = 0; i < count; i += 1) {
        if (array[i] != i * i) {
            printf("Element %d expected: %d actual %d", i, i *i, array[i]);
            isDataCorrect = false;
        }
    }
    if (isDataCorrect) printf("Data is correct\n");
}

int main()
{
    const int count = 1000000;
    int *hostArray;
    int *deviceArray;
    int size = count * sizeof(int);

    hostArray = new int[count];

    initializeData(hostArray, count);

    cudaMalloc(&deviceArray, size);
    cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice);

    resetData(hostArray, count);

    launchMaxOccupancyKernel(deviceArray, count);

    cudaMemcpy(hostArray, deviceArray, size, cudaMemcpyDeviceToHost);
    verifyData(hostArray, count);

    cudaFree(deviceArray);

    delete[] hostArray;

    return 0;
}

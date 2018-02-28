#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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
//Check addition CPU vs GPU
bool check_addition(float *arr_A,float *arr_B,float *arr_C,int elements);
//Check CUDA Errors
bool check (cudaError_t error );

# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from pycuda import driver, compiler, gpuarray
import time

# -- initialize the device
import pycuda.autoinit

USE_SIMPLE_KERNEL = 0
USE_TILED_KERNEL = 1

kernel_source_code = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{    
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    
    float Pvalue = 0;
    
    int full_size= %(MATRIX_SIZE)s*%(MATRIX_SIZE)s;
    
    if (ty < %(MATRIX_SIZE)s && tx < %(MATRIX_SIZE)s){             
        // Each thread loads one row of M and one column of N,
        //   to produce one element of P.
        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
            
            if(ty * %(MATRIX_SIZE)s + k < full_size &&  k * %(MATRIX_SIZE)s + tx < full_size){
                float Aelement = a[ty * %(MATRIX_SIZE)s + k];
                float Belement = b[k * %(MATRIX_SIZE)s + tx];
                Pvalue += Aelement * Belement;
            }
            
        }
        
        // Write the matrix to device memory;
        // each thread writes one element        
        c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
    }
}

__global__ void MatrixMulKernelTiled(float *A, float *B, float *C)
{
    
    const uint wA = %(MATRIX_SIZE)s;
    const uint wB = %(MATRIX_SIZE)s;
    
    // Block index
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    
    // Thread index
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    
    uint xIndex = tx + bx * tx ;
    
    
    //if (xIndex < wA){ 
        
        // Index of the first sub-matrix of A processed by the block
        const uint aBegin = wA * %(BLOCK_SIZE)s * by;
        // Index of the last sub-matrix of A processed by the block
        const uint aEnd = aBegin + wA - 1;
        // Step size used to iterate through the sub-matrices of A
        const uint aStep = %(BLOCK_SIZE)s;
        
        // Index of the first sub-matrix of B processed by the block
        const uint bBegin = %(BLOCK_SIZE)s * bx;
        // Step size used to iterate through the sub-matrices of B
        const uint bStep = %(BLOCK_SIZE)s * wB;
        
        // The element of the block sub-matrix that is computed
        // by the thread
        float Csub = 0;
        // Loop over all the sub-matrices of A and B required to
        // compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
        {
            // Shared memory for the sub-matrix of A
            __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
            // Shared memory for the sub-matrix of B
            __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
            
            // Load the matrices from global memory to shared memory
            // each thread loads one element of each matrix
            As[ty][tx] = A[a + wA * ty + tx];
            Bs[ty][tx] = B[b + wB * ty + tx];
            // Synchronize to make sure the matrices are loaded
            __syncthreads();
            
            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
            for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
                Csub += As[ty][k] * Bs[k][tx];
            
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }
        
        // Write the block sub-matrix to global memory;
        // each thread writes one element
        const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
        C[c + wB * ty + tx] = Csub;
    //}
}

"""


def cpu_operation(matrix_a, matrix_b):
    return np.dot(matrix_a, matrix_b)


def gpu_operation(matrix_a, matrix_b, results_gpu, kernel_binary, grid, blocks):
    kernel_binary(
        # inputs
        matrix_a, matrix_b,
        # output
        results_gpu,
        block=blocks,
        grid=grid
    )

    return results_gpu


def print_device_properties(dev):
    MAX_BLOCK_DIM_X = dev.get_attributes()[2]  # 1024
    MAX_BLOCK_DIM_Y = dev.get_attributes()[3]  # 1024
    MAX_BLOCK_DIM_Z = dev.get_attributes()[4]  # 64

    MAX_GRID_DIM_X = dev.get_attributes()[5]  # 2147483647
    MAX_GRID_DIM_Y = dev.get_attributes()[6]  # 65535
    MAX_GRID_DIM_Z = dev.get_attributes()[7]  # 65535

    MAX_THREAD_PER_BLOCK = dev.get_attributes()[1]  # 1024

    print('Device attributes: *******************************')
    print('MAX_BLOCK_DIM_X=', MAX_BLOCK_DIM_X)
    print('MAX_BLOCK_DIM_Y=', MAX_BLOCK_DIM_Y)
    print('MAX_BLOCK_DIM_Z=', MAX_BLOCK_DIM_Z)

    print('MAX_GRID_DIM_X=', MAX_GRID_DIM_X)
    print('MAX_GRID_DIM_Y=', MAX_GRID_DIM_Y)
    print('MAX_GRID_DIM_Z=', MAX_GRID_DIM_Z)
    print('MAX_THREAD_PER_BLOCK=', MAX_THREAD_PER_BLOCK)
    print('*' * 50)


def gpu_compile_kernel(kernel_type, matrix_size):
    driver.init()
    dev = driver.Device(0)

    # print_device_properties(dev)
    MAX_THREAD_PER_BLOCK = dev.get_attributes()[1]  # 1024

    threads_per_block = int(np.sqrt(MAX_THREAD_PER_BLOCK))
    number_of_blocks = int(matrix_size / threads_per_block)

    # check if a new tile is required
    if (number_of_blocks * threads_per_block) < matrix_size:
        number_of_blocks = number_of_blocks + 1

    print('## Kernel variables: ******************************')
    print('matriz size = ', matrix_size)
    print('threads per block = ', threads_per_block)
    print('number of blocks = ', number_of_blocks)
    print('*' * 50)

    grid = (threads_per_block, threads_per_block, 1)

    blocks = (number_of_blocks, number_of_blocks, 1)

    kernel_code = kernel_source_code % {'MATRIX_SIZE': matrix_size, 'BLOCK_SIZE': threads_per_block}

    # get the kernel function from the compiled module
    # compile the kernel code
    compiled_kernel = compiler.SourceModule(kernel_code)

    binary_gpu = None
    if kernel_type == USE_TILED_KERNEL:
        binary_gpu = compiled_kernel.get_function("MatrixMulKernelTiled")

    if kernel_type == USE_SIMPLE_KERNEL:
        binary_gpu = compiled_kernel.get_function("MatrixMulKernel")

    return binary_gpu, grid, blocks


def compare_results(time_cpu, time_gpu, c_cpu, c_gpu, blocks, grid):
    print('## Results: **************************************')
    print('Time CPU %10.8f' % time_cpu)
    print('Time GPU %10.8f' % time_gpu)
    print("Speedup: %5.4f" % (time_cpu / time_gpu))

    # check errors
    error = np.amax(c_cpu - c_gpu.get())
    if error < ERROR_THRESHOLD:
        print('SIZE:', matrix_size, 'SUCCESS - max difference: ', error)
    else:
        print('SIZE:', matrix_size, '* ERROR above threshold * - max difference: ', error)

    print("Blocks: ", blocks)
    print("Grid: ", grid)


def compare_matrix_operations(matrix_size):
    # create two random square matrices
    a_cpu = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    b_cpu = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    c_cpu = np.empty((matrix_size, matrix_size), np.float32)

    # operation using the CPU
    tic = time.time()
    c_cpu = cpu_operation(a_cpu, b_cpu)
    time_cpu = time.time() - tic

    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((matrix_size, matrix_size), np.float32)

    # compile kernel
    print("## Simple kernel GPU operation ########################################")
    kernel_binary, grid, blocks = gpu_compile_kernel(USE_SIMPLE_KERNEL, matrix_size)

    # operation using the GPU
    tic = time.time()
    # call the kernel on the card
    c_gpu = gpu_operation(a_gpu, b_gpu, c_gpu, kernel_binary, blocks, grid)
    time_gpu = time.time() - tic  # time measure

    compare_results(time_cpu, time_gpu, c_cpu, c_gpu, blocks, grid)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((matrix_size, matrix_size), np.float32)

    # compile kernel
    print("## Tiled kernel GPU operation ########################################")
    kernel_binary, grid, blocks = gpu_compile_kernel(USE_TILED_KERNEL, matrix_size)

    # operation using the GPU
    tic = time.time()
    # call the kernel on the card
    c_gpu = gpu_operation(a_gpu, b_gpu, c_gpu, kernel_binary, blocks, grid)
    time_gpu = time.time() - tic  # time measure

    compare_results(time_cpu, time_gpu, c_cpu, c_gpu, blocks, grid)


if __name__ == "__main__":
    ERROR_THRESHOLD = 0.001
    MAX_MATRIX_SIZE = 3000

    matrix_size = 4

    while matrix_size <= MAX_MATRIX_SIZE:
        compare_matrix_operations(matrix_size)
        matrix_size = matrix_size * 2

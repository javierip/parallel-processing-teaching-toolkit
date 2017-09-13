# -*- coding: utf-8 -*-
"""
Multiplies two square matrices together using a *single* block of threads and 
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import time
# -- initialize the device
import pycuda.autoinit


kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N,
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}

__global__ void MatrixMulKernel2(float *A, float *B, float *C)
{

  const uint wA = %(MATRIX_SIZE)s;
  const uint wB = %(MATRIX_SIZE)s;

  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

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
}
"""


def cpu_operation(matrix_a ,matrix_b):

    return np.dot(matrix_a , matrix_b)

def gpu_operation(matrix_a,matrix_b, results_gpu,binary_kernel, MATRIX_SIZE):

    # call the kernel on the card   
    
    driver.init()
    dev = driver.Device(0) 

    MAX_BLOCK_DIM_X=dev.get_attributes()[2] #1024
    MAX_BLOCK_DIM_Y=dev.get_attributes()[3] #1024
    MAX_BLOCK_DIM_Z=dev.get_attributes()[4] #64
    
    MAX_GRID_DIM_X=dev.get_attributes()[5] #2147483647
    MAX_GRID_DIM_Y=dev.get_attributes()[6] #65535
    MAX_GRID_DIM_Z=dev.get_attributes()[7] #65535

    print '*' * 50
    print 'MAX_BLOCK_DIM_X=',MAX_BLOCK_DIM_X
    print 'MAX_BLOCK_DIM_Y=',MAX_BLOCK_DIM_Y
    print 'MAX_BLOCK_DIM_Z=',MAX_BLOCK_DIM_Z

    print 'MAX_GRID_DIM_X=' ,MAX_GRID_DIM_X
    print 'MAX_GRID_DIM_Y=' ,MAX_GRID_DIM_Y
    print 'MAX_GRID_DIM_Z=' ,MAX_GRID_DIM_Z
    print '*' * 50

    #pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK
    """
    bdim = (MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, 1)

    dx, mx = divmod(MATRIX_SIZE, bdim[0])
    dy, my = divmod(MATRIX_SIZE, bdim[1])
   
    gdim = (dx+1,dy+1)
    """

    bdim = (1024, 1024, 1)
    dx, mx = divmod(MATRIX_SIZE, bdim[0])
    dy, my = divmod(MATRIX_SIZE, bdim[1])

    gdim = ( (dx + (mx>0)) * bdim[0]), ((dy + (my>0)) * bdim[1])
    #gdim = ( (dx + (mx>0)) ), (dy + (my>0))  

    #gdim=(1,1)

    print 'BLOCK=', bdim
    print 'GRID=', gdim
    print 'GRID SIZE X=' , gdim[0]*bdim[0]
    print '*' * 50
    


    binary_kernel(
    # inputs
    matrix_a , matrix_b,
    # output
    results_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    #block=(MATRIX_SIZE, MATRIX_SIZE, 1), #Colums , Rows , 1 
    block = bdim,
    grid = gdim
    )

    return results_gpu

def input_matrix_size():

    print "Enter the length of matrix N*N"

    length=input()


    return length 
        

def compare_cpu_vs_gpu_operations(MATRIX_SIZE):

    # define the (square) matrix size
    #  note that we'll only use *one* block of threads here
    #  as a consequence this number (squared) can't exceed max_threads,
    #  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
    #  for more information on how to get this number for your device
    #MATRIX_SIZE=32 #max

    # create two random square matrices
    a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    c_cpu = np.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # compute reference on the CPU to verify GPU computation


    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

    # get the kernel code from the template 
    # by specifying the constant MATRIX_SIZE

    BLOCK_SIZE = 2

    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'BLOCK_SIZE': BLOCK_SIZE,
        
    }

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    binary_gpu = mod.get_function("MatrixMulKernel")

    #operation using the GPU
    tic=time.time() 
    # call the kernel on the card
    c_gpu=gpu_operation(a_gpu,b_gpu, c_gpu,binary_gpu, MATRIX_SIZE)
    """
    binary_gpu(
    # inputs
    a_gpu , b_gpu,
    # output
    c_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block=(MATRIX_SIZE, MATRIX_SIZE, 1), #Colums , Rows , 1 

    )
    """

    time_gpu=time.time()-tic #time measure

    #operation using the CPU
    tic=time.time()

    c_cpu = cpu_operation(a_cpu,b_cpu)

    time_cpu=time.time()-tic

    # print the results
    print "-" * 80
    print "CPU-GPU difference:"
    print c_cpu - c_gpu.get()
    if np.allclose(c_cpu,c_gpu.get()):
        print 'SUCCESS'
    else:
        print '* ERROR *'

    print "-" * 80
    print "Time CPU:" , time_cpu
    print "Time GPU:" , time_gpu

    np.allclose(c_cpu, c_gpu.get())

if __name__ == "__main__":

   # while 1:
        #size= input_matrix_size()
        #size =int(size)
        size=33
        #length=8000
        compare_cpu_vs_gpu_operations(size)
# -*- coding: utf-8 -*-

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import time
# -- initialize the device
import pycuda.autoinit


kernel_code_template = """
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

__global__ void MatrixMulKernelTilled(float *A, float *B, float *C)
{

  const uint wA = %(MATRIX_SIZE)s;
  const uint wB = %(MATRIX_SIZE)s;

  // Block index
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  // Thread index
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;


  int txx = threadIdx.x + blockIdx.x * blockDim.x;
  int tyy = threadIdx.y + blockIdx.y * blockDim.y;

  

  if (tyy < %(MATRIX_SIZE)s && txx < %(MATRIX_SIZE)s)

  {

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
      for (int k = 0; k < %(BLOCK_SIZE)s, k <%(MATRIX_SIZE)s ; ++k)
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

}

"""


def cpu_operation(matrix_a ,matrix_b):

    return np.dot(matrix_a , matrix_b)

def gpu_operation(matrix_a,matrix_b, results_gpu,flag_tilled, MATRIX_SIZE):

     
    driver.init()
    dev = driver.Device(0) 

    MAX_BLOCK_DIM_X=dev.get_attributes()[2] #1024
    MAX_BLOCK_DIM_Y=dev.get_attributes()[3] #1024
    MAX_BLOCK_DIM_Z=dev.get_attributes()[4] #64
    
    MAX_GRID_DIM_X=dev.get_attributes()[5] #2147483647
    MAX_GRID_DIM_Y=dev.get_attributes()[6] #65535
    MAX_GRID_DIM_Z=dev.get_attributes()[7] #65535

    MAX_THREAD_PER_BLOCK=dev.get_attributes()[1] #1024

    PRINT_SPECS=0
    if PRINT_SPECS:
      print '*' * 50
      print 'MAX_BLOCK_DIM_X=',MAX_BLOCK_DIM_X
      print 'MAX_BLOCK_DIM_Y=',MAX_BLOCK_DIM_Y
      print 'MAX_BLOCK_DIM_Z=',MAX_BLOCK_DIM_Z

      print 'MAX_GRID_DIM_X=' ,MAX_GRID_DIM_X
      print 'MAX_GRID_DIM_Y=' ,MAX_GRID_DIM_Y
      print 'MAX_GRID_DIM_Z=' ,MAX_GRID_DIM_Z
      print 'MAX_THREAD_PER_BLOCK=' ,MAX_THREAD_PER_BLOCK
      print '*' * 50


    THREADS=int(np.sqrt(MAX_THREAD_PER_BLOCK))

    bdim = (THREADS, THREADS, 1)


    dx, mx = divmod(MATRIX_SIZE-1, bdim[0])
    dy, my = divmod(MATRIX_SIZE-1, bdim[1])

 

    gdim = (dx+1,dy+1,1)

   
    
    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE,
        'BLOCK_SIZE': bdim[0],
        
    }

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    
    if flag_tilled:
      binary_gpu = mod.get_function("MatrixMulKernelTilled")
      print "tilled kernel"

    else:
      binary_gpu = mod.get_function("MatrixMulKernel")

    binary_gpu(
    # inputs
    matrix_a , matrix_b,
    # output
    results_gpu,
    block = bdim,
    grid = gdim
    )

    return results_gpu , bdim, gdim

def input_matrix_size():

    print "Enter the length of matrix N*N"

    length=input()


    return length 
        

def compare_cpu_vs_gpu_operations(MATRIX_SIZE,flag_tilled):

    # define the (square) matrix size
    #  note that we'll only use *one* block of threads here
    #  as a consequence this number (squared) can't exceed max_threads,
    #  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
    #  for more information on how to get this number for your device
 

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

     

    #operation using the GPU
    tic=time.time() 
    # call the kernel on the card
    c_gpu,bdim,gdim=gpu_operation(a_gpu,b_gpu, c_gpu,flag_tilled, MATRIX_SIZE)
   

    time_gpu=time.time()-tic #time measure

    #operation using the CPU
    tic=time.time()

    c_cpu = cpu_operation(a_cpu,b_cpu)

    time_cpu=time.time()-tic

    
    error_param = 1e-5
    error = np.amax(c_cpu - c_gpu.get())

    
    if error < error_param:
        print 'SIZE:', MATRIX_SIZE , 'SUCCESS', ' - error: ', error, bdim,gdim
    else:
        print 'SIZE:', MATRIX_SIZE , '* ERROR **************************************************'
        print 'error: ', error


    return error

if __name__ == "__main__":

    flag_tilled=0

    for i in range (4,129):

      compare_cpu_vs_gpu_operations(i,flag_tilled)
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
__global__ void MatrixMulKernel(double *a, double *b, int size , double *c)
{

    // 2D Thread ID (assuming that only *one* block will be executed)

    //tid = threadIdx.x + blockIdx.x * blockDim.x;
    //tid += blockDim.x * gridDim.x;
    
    int tx = threadIdx.x + blockIdx.x * blockDim.x;    
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    //if(tx < size && ty < size)
    {


        // Pvalue is used to store the element of the matrix
        // that is computed by the thread
        double Pvalue = 0;

        // Each thread loads one row of M and one column of N,
        //   to produce one element of P.
        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
            double Aelement = a[ty * %(MATRIX_SIZE)s + k];
            double Belement = b[k * %(MATRIX_SIZE)s + tx];
            Pvalue += Aelement * Belement;
        }

        // Write the matrix to device memory;
        // each thread writes one element
        c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
        
    }
}
"""


# execute multiplication in CPU
def cpu_operation(matrix_a,matrix_b, matrix_c):

    matrix_c = np.dot(matrix_a, matrix_b)


def get_device_info():

    driver.init()
    dev = driver.Device(0) 

    MAX_THREADS_PER_BLOCK=dev.get_attributes()[2] #1024

    MAX_BLOCK_DIM=[]

    MAX_BLOCK_DIM.append(dev.get_attributes()[2]) #1024
    MAX_BLOCK_DIM.append(dev.get_attributes()[3]) #1024
    MAX_BLOCK_DIM.append(dev.get_attributes()[4]) #64
    
    MAX_GRID_DIM=[]

    MAX_GRID_DIM.append(dev.get_attributes()[5]) #2147483647
    MAX_GRID_DIM.append(dev.get_attributes()[6]) #65535
    MAX_GRID_DIM.append(dev.get_attributes()[7]) #65535

    MAX_SHARED_MEMORY_PER_BLOCK=dev.get_attributes()[8] #49152

    print '*' * 50
    print 'Running on: ' , dev.name()
    print 'MAX_THREADS_PER_BLOCK=',MAX_THREADS_PER_BLOCK
    
    print 'MAX_BLOCK_DIM_X=',MAX_BLOCK_DIM[0]
    print 'MAX_BLOCK_DIM_Y=',MAX_BLOCK_DIM[1]
    print 'MAX_BLOCK_DIM_Z=',MAX_BLOCK_DIM[2]

    print 'MAX_GRID_DIM_X=' ,MAX_GRID_DIM[0]
    print 'MAX_GRID_DIM_Y=' ,MAX_GRID_DIM[1]
    print 'MAX_GRID_DIM_Z=' ,MAX_GRID_DIM[2]
    
    print 'MAX_SHARED_MEMORY_PER_BLOCK=' ,MAX_SHARED_MEMORY_PER_BLOCK

    print '*' * 50

    return MAX_THREADS_PER_BLOCK,MAX_BLOCK_DIM,MAX_GRID_DIM,MAX_SHARED_MEMORY_PER_BLOCK

def compare_operations(length,threads):


    # define the (square) matrix size
    #  note that we'll only use *one* block of threads here
    #  as a consequence this number (squared) can't exceed max_threads,
    #  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
    #  for more information on how to get this number for your device
    MATRIX_SIZE = length  #max 32

    # create two random square matrices
    a_cpu = 25 * np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float64) +30
    b_cpu = 25 * np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float64) +30
    c_cpu = np.empty((MATRIX_SIZE, MATRIX_SIZE), np.float64)

    # compute reference on the CPU to verify GPU computation
    tic=time.time()
    #cpu_operation(a_cpu,b_cpu,c_cpu)
    c_cpu = np.dot(a_cpu, b_cpu)
    time_cpu=time.time()-tic

    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float64)

    # get the kernel code from the template 
    # by specifying the constant MATRIX_SIZE
    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
    }

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")


    tic=time.time() #time measure
    # call the kernel on the card
    print "MATRIX_SIZE ",MATRIX_SIZE

    print "threads ",threads

    GRID_SIZE=int(np.ceil(MATRIX_SIZE/np.sqrt(threads)))

    print "DENOM GRID SIZE " ,  np.sqrt(threads) 

    print "GRID_SIZE ",GRID_SIZE

    BLOCK_SIZE=int(MATRIX_SIZE//GRID_SIZE)

    print "BLOCK_SIZE ",BLOCK_SIZE


    matrixmul(
        # inputs
        a_gpu, b_gpu, int(MATRIX_SIZE),
        # output
        c_gpu,

        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block=(BLOCK_SIZE, BLOCK_SIZE, 1), #Colums , Rows , 1 
        grid=(GRID_SIZE,GRID_SIZE,1),


    )
    time_gpu=time.time()-tic #time measure
    # print the results
    print "-" * 80
    print "Matrix C (GPU):"
    print c_gpu.get()

    print "-" * 80
    print "CPU-GPU difference:"
    print c_cpu - c_gpu.get()



    print "-" * 80
    print "Time CPU:" , time_cpu
    print "Time GPU:" , time_gpu

    print "Error Max:", np.amax(c_cpu - c_gpu.get())
    print "Matrix Size:" ,MATRIX_SIZE, "*", MATRIX_SIZE, "=", MATRIX_SIZE*MATRIX_SIZE, " elements"
    print "BLOCK:", MATRIX_SIZE/GRID_SIZE,MATRIX_SIZE/GRID_SIZE,1
    print "GRID:", 1,1,1


    np.allclose(c_cpu, c_gpu.get())

def user_inputs(max_threads):

    print
    print '-' * 50
    N  = int(input("Define N dimension of the matrix: N * N = "))
    threads = int(input("Define number of threads per block = "))
    while(threads>max_threads):
        print "The threads efined value is greater than the maximum allowed by the architecture "
        threads = int(input("Define number of threads per block = "))
    return N, threads 

if __name__ == "__main__":



    MAX_THREADS_PER_BLOCK,MAX_BLOCK_DIM,MAX_GRID_DIM,MAX_SHARED_MEMORY_PER_BLOCK = get_device_info()

    #matrix_size, threads_per_block  =   user_inputs(MAX_THREADS_PER_BLOCK)

    #compare_operations(matrix_size,threads_per_block)
    compare_operations(1560,1024)



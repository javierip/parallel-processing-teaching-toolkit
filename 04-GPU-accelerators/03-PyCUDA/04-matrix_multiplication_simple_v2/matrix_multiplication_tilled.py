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
__global__ void MatrixMulKernel(double *A, double *B, double *C)
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
  double Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;  a <= aEnd-1; a += aStep, b += bStep)
    {
      // Shared memory for the sub-matrix of A

      __shared__ double As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ double Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

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

    // Shared memory for the sub-matrix of A //////////////////////////////

      __shared__ double As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ double Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

      // Load the matrices from global memory to shared memory
      // each thread loads one element of each matrix
	
     
      As[ty][tx] = A[aEnd + wA * ty + tx];
      Bs[ty][tx] = B[aEnd + wB * ty + tx];
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

	// Write the block sub-matrix to global memory;
	// each thread writes one element
	const uint c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
	
	if(c + wB * ty + tx<1444)
  	C[c + wB * ty + tx] = Csub;


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

   


	tic=time.time() #time measure
	# call the kernel on the card
	print "MATRIX_SIZE ",MATRIX_SIZE

	print "threads ",threads

	GRID_SIZE=int(np.ceil(float(MATRIX_SIZE)/np.sqrt(threads)))

	print (np.sqrt(threads))
	print "GRID_SIZE" , GRID_SIZE

	BLOCK_SIZE=int(np.ceil((float(MATRIX_SIZE)/GRID_SIZE)))

	BLOCK_SIZE=32


	while(BLOCK_SIZE>32): #32*32=1024 = MAX THREADS PER BLOCK FOR MY GPU
		print "BLOCK_SIZE>32"
		BLOCK_SIZE-=1



	print "BLOCK_SIZE ",BLOCK_SIZE

	# get the kernel code from the template 
	# by specifying the constant MATRIX_SIZE
	kernel_code = kernel_code_template % {
		'MATRIX_SIZE': MATRIX_SIZE,
		'BLOCK_SIZE': BLOCK_SIZE,
	}


	# compile the kernel code 
	mod = compiler.SourceModule(kernel_code)

	# get the kernel function from the compiled module
	matrixmul = mod.get_function("MatrixMulKernel")


	matrixmul(
		# inputs
		a_gpu, b_gpu,
		# output
		c_gpu,

		# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
		block=(BLOCK_SIZE, BLOCK_SIZE, 1), #Colums , Rows , 1 
		grid=(GRID_SIZE,GRID_SIZE,1),


	)
	time_gpu=time.time()-tic #time measure
	# print the results
	"""
	print "-" * 80
	print "CPU-GPU difference:"
	print c_cpu - c_gpu.get()
	"""


	print "-" * 80
	print "Time CPU:" , time_cpu
	print "Time GPU:" , time_gpu
	error = np.amax(c_cpu - c_gpu.get())
	print "Error Max:", error
	print "Matrix Size:" ,MATRIX_SIZE, "*", MATRIX_SIZE, "=", MATRIX_SIZE*MATRIX_SIZE, " elements"
	print "BLOCK:", BLOCK_SIZE,BLOCK_SIZE,1
	print "GRID:", GRID_SIZE,GRID_SIZE,1


	np.allclose(c_cpu, c_gpu.get())

	if abs(error)< 0.000001: print "CHECKED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

	return error

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



	#MAX_THREADS_PER_BLOCK,MAX_BLOCK_DIM,MAX_GRID_DIM,MAX_SHARED_MEMORY_PER_BLOCK = get_device_info()

	#matrix_size, threads_per_block  =   user_inputs(MAX_THREADS_PER_BLOCK)

	#compare_operations(matrix_size, threads_per_block)

	
	compare_operations(38,1024)





import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools
import numpy as np
import time


#Device Source
kernel_code_multi = """
	__global__ void kernel(float *m, float *a, float *b)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 m[tid] = a[tid] * b[tid];
	}
	"""


def cpu_operation(vector_a,vector_b, vector_res, VECTOR_LEN):

	for i in range(0,VECTOR_LEN):
		vector_res[i]=vector_a[i]*vector_b[i]

	return vector_res

# execute reduction in GPU
def gpu_operation(vector_a,vector_b, results_gpu,binary_kernel, VECTOR_LEN):


	#if dimensions==1:


	# call the kernel on the card	
	driver.init()
	dev = driver.Device(0) 

	MAX_THREADS_PER_BLOCK=dev.get_attributes()[2] #1024

	MAX_BLOCK_DIM_X=dev.get_attributes()[2] #1024
	MAX_BLOCK_DIM_Y=dev.get_attributes()[3] #1024
	MAX_BLOCK_DIM_Z=dev.get_attributes()[4] #64
	
	MAX_GRID_DIM_X=dev.get_attributes()[5] #2147483647
	MAX_GRID_DIM_Y=dev.get_attributes()[6] #65535
	MAX_GRID_DIM_Z=dev.get_attributes()[7] #65535

	MAX_SHARED_MEMORY_PER_BLOCK=dev.get_attributes()[8] #49152
	
	print dev.name()
	print 'MAX_THREADS_PER_BLOCK=',MAX_THREADS_PER_BLOCK
	print '*' * 50
	print 'MAX_BLOCK_DIM_X=',MAX_BLOCK_DIM_X
	print 'MAX_BLOCK_DIM_Y=',MAX_BLOCK_DIM_Y
	print 'MAX_BLOCK_DIM_Z=',MAX_BLOCK_DIM_Z

	print 'MAX_GRID_DIM_X=' ,MAX_GRID_DIM_X
	print 'MAX_GRID_DIM_Y=' ,MAX_GRID_DIM_Y
	print 'MAX_GRID_DIM_Z=' ,MAX_GRID_DIM_Z
	print '*' * 50
	print 'MAX_SHARED_MEMORY_PER_BLOCK=' ,MAX_SHARED_MEMORY_PER_BLOCK
	print '*' * 50
	#attrs.iteritems()
	
	#pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK
	bdim = (MAX_BLOCK_DIM_X, 1, 1)


	dx, mx = divmod(VECTOR_LEN, bdim[0])
	#dy, my = divmod(1, bdim[1])
	

	gdim = (dx+1,1)

	#gdim = ( (dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1]) 

	#print 'dx=' ,dx , ' mx=', mx

	print 'BLOCK=', bdim
	print 'GRID=', gdim
	print 'GRID SIZE X=' , gdim[0]*bdim[0]

	print '*' * 50

	binary_kernel(
		
		# output
		results_gpu,
		# inputs
		vector_a,		
		vector_b,
		# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
		block = bdim,
		grid = (dx+1,1)
		)


	return results_gpu


def input_vector_len():

	print "Enter the length of the vector"

	length=input()

	return length;
		


def compare_cpu_vs_gpu_operations(length):

	VECTOR_LEN = length

	#Host variables
	a_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)
	b_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)


	res_cpu =  np.zeros(VECTOR_LEN, np.float32)

	# transfer host (CPU) memory to device (GPU) memory
	a_gpu = gpuarray.to_gpu(a_cpu)
	b_gpu = gpuarray.to_gpu(b_cpu)

	res_gpu = gpuarray.empty((VECTOR_LEN), np.float32)

	# compile the kernel code
	# select the kernel from user console and
	# get the kernel function from the compiled module 



	# get the kernel code from the template
	# by specifying the constant VECTOR_LEN
	#Your need define a gpu_function like a kernel previously
	#kernel_code = kernel_code_addtion % {'VECTOR_LEN': VECTOR_LEN}	



	kernel_code = kernel_code_multi % {'VECTOR_LEN': VECTOR_LEN}	
		

	
	# compile the kernel code
	mod = compiler.SourceModule(kernel_code)

	 # get the kernel function from the compiled module
	reduction_binary_gpu = mod.get_function("kernel")

	#operation using the GPU
	tic=time.time()

	res_gpu = gpu_operation(a_gpu, b_gpu, res_gpu, reduction_binary_gpu, VECTOR_LEN)

	time_gpu=time.time()-tic

	#operation using the CPU
	tic = time.time()
	
	res_cpu=cpu_operation(a_cpu,b_cpu, res_cpu, VECTOR_LEN)

	time_cpu = time.time() - tic


	#print results

	print "Vector Operation A*B"	


	print "Vector Size:", VECTOR_LEN
	print "-" * 80
	print "GPU-CPU Diference" , res_cpu-res_gpu.get()
	if np.allclose(res_cpu,res_gpu.get()):
		print 'SUCCESS'
	else:
		print '* ERROR *'

	print "Time CPU:", time_cpu
	print "Time GPU:", time_gpu






if __name__ == "__main__":

	while 1:
		length=input_vector_len()
	
		#length=8000
		compare_cpu_vs_gpu_operations(length)









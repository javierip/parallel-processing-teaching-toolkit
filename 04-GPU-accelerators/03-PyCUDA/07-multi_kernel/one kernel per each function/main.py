import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools
import numpy as np
import time


#Device Source
kernel_code_addtion = """
	__global__ void kernel(float *s, float *a, float *b)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 s[tid] = a[tid] + b[tid];
	}
	"""
kernel_code_multi = """
	__global__ void kernel(float *m, float *a, float *b)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 m[tid] = a[tid] * b[tid];
	}
	"""
kernel_code_div = """
		__global__ void kernel(float *d, float *a, float *b)
	{
		  int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 d[tid] = a[tid] / b[tid];
	}
"""

def cpu_operation(vector_a,vector_b, vector_res, VECTOR_LEN, option):
	if(option==1):
		for i in range(0,VECTOR_LEN):
			vector_res[i]=vector_a[i]+vector_b[i]
	elif (option==2):
		for i in range(0,VECTOR_LEN):
			vector_res[i]=vector_a[i]*vector_b[i]
	elif (option==3):    
		for i in range(0,VECTOR_LEN):
			vector_res[i]=vector_a[i]/vector_b[i]
	return vector_res

# execute reduction in GPU
def gpu_operation(vector_a,vector_b, results_gpu,binary_kernel, VECTOR_LEN):

	# call the kernel on the card	
	binary_kernel(
		
		# output
		results_gpu,
		# inputs
		vector_a,		
		vector_b,
		# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
		block = (VECTOR_LEN, 1, 1),
		)


	return results_gpu

def select_operation():
	option=0
	while(option<1 or option>3):
		print "What operation do you want to perform?"
		print "1.Addition"
		print "2.Multiplication"
		print "3.Divition"

		option=input()
		if(option<1 or option>3):	
			print "Invalid Option"

	return option

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

	option=select_operation()
	option =1
	
	if(option==1):	
		# get the kernel code from the template
		# by specifying the constant VECTOR_LEN
		kernel_code = kernel_code_addtion % {'VECTOR_LEN': VECTOR_LEN}	
		
	elif (option==2):
		kernel_code = kernel_code_multi % {'VECTOR_LEN': VECTOR_LEN}	
		
	elif (option==3):    
		kernel_code = kernel_code_div % {'VECTOR_LEN': VECTOR_LEN}	
		
	else:
		print "Invalid Option"
	
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
	
	res_cpu=cpu_operation(a_cpu,b_cpu, res_cpu, VECTOR_LEN, option)

	time_cpu = time.time() - tic


	#print results
	if(option==1):
		print "Vector Operation A+B"	
	elif (option==2):
		print "Vector Operation A*B"	
	elif (option==3):   
		print "Vector Operation A/B"

	print "Vector Size:", VECTOR_LEN
	print "-" * 80
	print "GPU-CPU Diference" , res_cpu-res_gpu.get()
	print "Time CPU:", time_cpu
	print "Time GPU:", time_gpu






if __name__ == "__main__":
	
	compare_cpu_vs_gpu_operations(1024)









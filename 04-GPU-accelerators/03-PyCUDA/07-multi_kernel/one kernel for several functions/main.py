import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import  gpuarray
import numpy as np
import time


#Device Source
kernel_code = SourceModule("""
    __global__ void Add(float *s, float *a, float *b)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

         s[tid] = a[tid] + b[tid];
    }

    __global__ void Multi(float *m, float *a, float *b)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

         m[tid] = a[tid] * b[tid];
    }

        __global__ void Div(float *d, float *a, float *b)
    {
          int tid = blockIdx.x * blockDim.x + threadIdx.x;

         d[tid] = a[tid] / b[tid];
    }
""")

def cpu_operation(vector_a,vector_b,vector_c,vector_d, vector_res, VECTOR_SIZE):
    res_cpu_1 =  np.zeros(VECTOR_SIZE, np.float32)
    res_cpu_2 =  np.zeros(VECTOR_SIZE, np.float32)
    
    for i in range(0,VECTOR_SIZE):
        res_cpu_1[i]=vector_a[i]*vector_b[i]
    
    for i in range(0,VECTOR_SIZE):
        res_cpu_2[i]=vector_c[i]/vector_d[i]
    
    for i in range(0,VECTOR_SIZE):
        vector_res[i]=res_cpu_1[i]+res_cpu_2[i]
    
    return vector_res

def gpu_operation(vector_a,vector_b,vector_c,vector_d, vector_res, gpu_add_binary ,gpu_multi_binary,gpu_div_binary, VECTOR_SIZE):    
    res_gpu_1 = gpuarray.empty((VECTOR_SIZE), np.float32)
    res_gpu_2 = gpuarray.empty((VECTOR_SIZE), np.float32)
    
    res_cpu_1 =  np.zeros(VECTOR_SIZE, np.float32)
    res_cpu_2 =  np.zeros(VECTOR_SIZE, np.float32)
  
    gpu_multi_binary(
        # output
        res_gpu_1, 
        # inputs
        vector_a,        
        vector_b,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (VECTOR_SIZE, 1, 1),
        )
   

    gpu_div_binary(
        # output
        res_gpu_2,
        # inputs
        vector_c,
        vector_d,
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (VECTOR_SIZE, 1, 1),
        )

    
    gpu_add_binary(
        # output
        vector_res,
        # inputs
        res_gpu_1,  
        res_gpu_2, 
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (VECTOR_SIZE, 1, 1),
        )

    
    return vector_res


def compare_cpu_vs_gpu_operations(length):

    VECTOR_LEN = length
    #Host variables
    a_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)
    b_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)
    c_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)
    d_cpu =  np.random.randint(1,10,size=VECTOR_LEN).astype(np.float32)


    res_cpu =  np.zeros(VECTOR_LEN, np.float32)

    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.to_gpu(c_cpu)
    d_gpu = gpuarray.to_gpu(d_cpu)


    res_gpu = gpuarray.empty((VECTOR_LEN), np.float32)

    # compile the kernel code

    # get the kernel function from the compiled module
    gpu_multi = kernel_code.get_function("Multi")
    gpu_div = kernel_code.get_function("Div")
    gpu_add = kernel_code.get_function("Add")

    # call the kernel on the card
    tic=time.time()
    
    res_gpu=gpu_operation(a_gpu ,b_gpu,c_gpu,d_gpu, res_gpu, gpu_add,gpu_multi,gpu_div, VECTOR_LEN)

    time_gpu=time.time()-tic


    tic=time.time()
    #Operation using the cpu
    res_cpu=cpu_operation(a_cpu,b_cpu,c_cpu,d_cpu, res_cpu, VECTOR_LEN)

    time_cpu=time.time()-tic
    #Vector multiplication by constant

    print "Vector Operation (A/B+C*B)"
    print "Vector Size:", VECTOR_LEN
    print "-" * 80
    print "GPU-CPU Diference" , res_cpu-res_gpu.get()
    print "Time CPU:", time_cpu
    print "Time GPU:", time_gpu

if __name__ == "__main__":
    
    compare_cpu_vs_gpu_operations(1024)

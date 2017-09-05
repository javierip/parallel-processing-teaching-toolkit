import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void vectorReduce(volatile float *global_input_data, volatile float *global_output_data)
{


    __shared__ float sdata[%(VECTOR_LEN)s];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x ) + threadIdx.x;
    sdata[tid] = global_input_data[i];


    __syncthreads();


    int s = blockDim.x / 2; 

    while(s>1){


        __syncthreads();

         if (tid < s ) {
        
                sdata[tid] += sdata[tid + s];
            

            
            __syncthreads();
        }      

        if(s%%2==0)
            s=s/2;
        else
            s=(s+1)/2;

        __syncthreads();

        if(s<=0)break;

    }

     __syncthreads();

     if (tid == 0){
        sdata[0]+=sdata[1];
    }   
             
    

    if (tid == 0) {
        *global_output_data = sdata[0];

    }

 
}
"""


# excecute reduction in CPU
def cpu_reduction(vector_cpu, VECTOR_LEN):
    sum_value = 0

    for i in range(0, VECTOR_LEN):
        sum_value += vector_cpu[i]


    return sum_value

# execute reduction in CPU
def gpu_reduction(vector_gpu, results_gpu, reduction_binary_gpu, VECTOR_LEN):
    reduction_binary_gpu(
        # inputs
        vector_gpu,
        # output
        results_gpu,
        # (only one) block of VECTOR_LEN x VECTOR_LEN threads
        block=(VECTOR_LEN, 1, 1),
    )
    return results_gpu


def compare_reduction_operations(length):
    VECTOR_LEN = length

    # create a vector of random float numbers
    vector_cpu = np.random.randn(VECTOR_LEN).astype(np.float32)

    # get the kernel code from the template
    # by specifying the constant VECTOR_LEN
    kernel_code = kernel_code_template % {'VECTOR_LEN': VECTOR_LEN}

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    reduction_binary_gpu = mod.get_function("vectorReduce")

    # CPU reduction
    tic = time.time()
    result = cpu_reduction(vector_cpu, VECTOR_LEN)

    time_cpu = time.time() - tic

    print "-" * 80
    print 'Result CPU:', result
    print "Time CPU:", time_cpu
    print "-" * 80

    # GPU reduction
    tic = time.time()
    vector_gpu = gpuarray.to_gpu(vector_cpu)
    results_gpu = gpuarray.empty((1), np.float32)
    result = gpu_reduction(vector_gpu, results_gpu, reduction_binary_gpu, VECTOR_LEN)
    time_gpu = time.time() - tic

    print 'Result GPU:', result
    print "Time GPU:", time_gpu
    print "-" * 80


if __name__ == "__main__":
    compare_reduction_operations(1024)

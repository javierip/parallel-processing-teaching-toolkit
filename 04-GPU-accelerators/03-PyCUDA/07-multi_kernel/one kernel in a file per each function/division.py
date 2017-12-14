#Device Source for vector division

kernel_code_div = """
		__global__ void kernel(float *d, float *a, float *b)
	{
		  int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 d[tid] = a[tid] / b[tid];
	}
"""
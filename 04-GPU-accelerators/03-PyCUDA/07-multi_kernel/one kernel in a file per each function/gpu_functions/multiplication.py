#Device Source for vector multiplication

kernel_code_multi = """
	__global__ void kernel(float *m, float *a, float *b)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 m[tid] = a[tid] * b[tid];
	}
	"""

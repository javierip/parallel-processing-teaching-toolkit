#Device Source for vector addition

kernel_code_addition = """
	__global__ void kernel(float *s, float *a, float *b)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		 s[tid] = a[tid] + b[tid];
	}
	"""

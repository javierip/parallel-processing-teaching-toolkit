
__kernel void vector_multi (__global const float *a_gpu, __global const float *b_gpu, __global float *res_gpu) {
	


	int tx = get_global_id(0);
	int ty = get_global_id(1);

		
     /*
    gridDim 	=	get_num_groups()
	blockDim 	=	get_local_size()
	blockIdx 	=	get_group_id()
	threadIdx 	=	get_local_id()
	blockIdx * blockDim + threadIdx 	=	get_global_id()
	gridDim * blockDim 					=	get_global_size() 
    */

	int ssize=32; 


	int k=0;
	for(k=0;k<ssize;k++){

		int a_element = a_gpu[tx+ty*ssize+k];
		int b_element =  b_gpu[tx+ty*ssize+k];
		res_gpu[tx+ty*ssize+k]= a_element + b_element;


		
	}
	
}

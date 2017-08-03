//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>   /* ceil */

#define  N_VECTORS 3  //Number of arrays needed
//  Constants
const int VECTOR_SIZE = 32;
const int MATRIX_SIZE = VECTOR_SIZE*VECTOR_SIZE;
///More cool fuction set by flash

//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
cl_context cl_CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}
//  Create memory objects used as the arguments to the kernel
bool cl_CreateMemObject(cl_context context, cl_mem memObject, int size, float *arr[], int flag_rw) {

    if (flag_rw)
        memObject = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size*size, arr, NULL);

    else
        memObject = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size*size, arr, NULL);

    if (memObject == NULL) {
        std::cerr << "Error creating memory objects." << std::endl;

        return false;
    }

    return true;


}
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * MATRIX_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * MATRIX_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * MATRIX_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}
//  Create a command queue on the first device available on the
//  context
cl_command_queue cl_CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}




void cl_cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem *memObjects, int n_objetcs)
{
    for (int i = 0; i < n_objetcs; i++)
        if (memObjects[i] != NULL)
            clReleaseMemObject(memObjects[i]);

    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

//  Create an OpenCL program from the kernel source file
cl_program cl_CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}



bool cpu_check(float **arr_a, float **arr_b,  int n) {

    for (int i = 0; i < n; i++)
        if (arr_b[i] != arr_a[i])
            return 0;
    return 1;
}
//  main() for HelloWorld example
int main(int argc, char** argv)
{
    //CL variables
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;

    clock_t start, end;


    // Create an OpenCL context on first available platform
    context = cl_CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = cl_CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = cl_CreateProgram(context, device, "kernel.cl");
    if (program == NULL)
    {
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "vector_multi", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[VECTOR_SIZE][VECTOR_SIZE];
    float result_cpu[VECTOR_SIZE][VECTOR_SIZE];
    float a[VECTOR_SIZE][VECTOR_SIZE];
    float b[VECTOR_SIZE][VECTOR_SIZE];
    for (int j = 0; j < VECTOR_SIZE; j++)
    for (int i = 0; i < VECTOR_SIZE; i++)
        {
            a[i][j] = (float)i;
            b[i][j] = (float)(i);
        }
    //cl_createMemObject_float((cl_context context, cl_mem memObject, int size, float * arr, int flag_rw) {

    // Create 3 vectors like a

    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * VECTOR_SIZE*VECTOR_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * VECTOR_SIZE*VECTOR_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * VECTOR_SIZE*VECTOR_SIZE, result, NULL);

            printf("VECTOR A\n");
        for (int i = 0; i < VECTOR_SIZE; i++){
        for (int j = 0; j < VECTOR_SIZE; j++)
   
            printf("%0.1lf\t",a[i][j] );
        printf("\n");
        }
        printf("VECTOR B\n");
            for (int i = 0; i < VECTOR_SIZE; i++){
        for (int j = 0; j < VECTOR_SIZE; j++)
   
            printf("%0.1lf\t",b[i][j] );
        printf("\n");
        }
                printf("VECTOR R\n");


    if (memObjects[0] == NULL) 
        std::cerr << "Error creating memory objects." << std::endl;
     if (memObjects[1] == NULL) 
        std::cerr << "Error creating memory objects." << std::endl;
     if (memObjects[2] == NULL) 
        std::cerr << "Error creating memory objects." << std::endl;


    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }



    size_t globalWorkSize[1] = { 1024 };
    size_t localWorkSize[1] = { 16 };



    std::cout << "Matrix Addition with " << VECTOR_SIZE << "*" VECTOR_SIZE << " Elements" <<std::endl;

    start = clock();
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    end = clock();
    printf("Tiempo GPU: %lf\n", (double ) (end - start) / CLOCKS_PER_SEC * 1000);


    //CPU Operation for comparation
    start = clock();
    for (int j = 0; j < VECTOR_SIZE; j++)
        for (int i = 0; i < VECTOR_SIZE; i++)
            result_cpu[i][j] = a[i][j] + b[i][j];
    end = clock();
    printf("Tiempo CPU: %lf\n", (double ) (end - start) / CLOCKS_PER_SEC * 1000);




    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, MATRIX_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
     for (int i = 0; i < VECTOR_SIZE; i++){
        for (int j = 0; j < VECTOR_SIZE; j++)
   
            printf("%0.1lf\t",result[i][j] );
        printf("\n");
        }
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }



    //Check result
    for (int i = 0; i < VECTOR_SIZE; i++)
        for (int j = 0; j < VECTOR_SIZE; j++)
            if (result_cpu[i][j] != result[i][j]){
                std::cout << "Something wrong" << std::endl;
                cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
                break;
            }
           
    std::cout << "Checked operation!" << std::endl;


    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;



 

    return 0;
}

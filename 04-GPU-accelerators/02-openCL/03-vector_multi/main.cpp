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

#define  N_VECTORS 3  //Number of arrays needed
//  Constants
const int VECTOR_SIZE = 100000;

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
bool cl_CreateMemObject(cl_context context, cl_mem memObject, int size, float *arr, int flag_rw) {

    if (flag_rw)
        memObject = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, arr, NULL);

    else
        memObject = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, arr, NULL);

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
                                   sizeof(float) * VECTOR_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * VECTOR_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * VECTOR_SIZE, NULL, NULL);

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

void cpu_multi (float *arr_a, float *arr_b, float *arr_c, int n) {


    for (int i = 0; i < n; i++)
        arr_c[i] = arr_a[i] * arr_b[i];

}


bool cpu_check(float *arr_a, float *arr_b,  int n) {

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
    float result[VECTOR_SIZE];
    float result_cpu[VECTOR_SIZE];
    float a[VECTOR_SIZE];
    float b[VECTOR_SIZE];
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i);
    }
    //cl_createMemObject_float((cl_context context, cl_mem memObject, int size, float * arr, int flag_rw) {

    // Create 3 vectors like a
    for (int i = 0; i < N_VECTORS; i++)
        if (!cl_CreateMemObject(context, memObjects[i], VECTOR_SIZE, a, 1))
        {
            cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
            return 1;
        }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }


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

    size_t globalWorkSize[1] = { VECTOR_SIZE };
    size_t localWorkSize[1] = { 1 };

    std::cout << "Vector Multiplication with " << VECTOR_SIZE << " Elements" <<std::endl;

    start = clock();
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    end = clock();
    printf("Tiempo GPU: %lf\n", (double ) (end - start) / CLOCKS_PER_SEC * 1000);


    //CPU Operation for comparation
    start = clock();
    cpu_multi(a, b, result_cpu, VECTOR_SIZE);
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
                                 0, VECTOR_SIZE * sizeof(float), result,
                                 0, NULL, NULL);


    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);
        return 1;
    }


    // Output the result buffer
    /*
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        std::cout << b[i] << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }

    std::cout << "\n";



    for (int i = 0; i < VECTOR_SIZE; i++)
    {
         std::cout << result_cpu[i] << " ";
    }*/

    if (cpu_check(result_cpu, result, VECTOR_SIZE))
        std::cout << "Checked operation!" << std::endl;
    else
        std::cout << "Something wrong" << std::endl;

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;



    cl_cleanup(context, commandQueue, program, kernel, memObjects, N_VECTORS);

    return 0;
}

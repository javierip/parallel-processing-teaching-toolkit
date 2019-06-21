cmake_minimum_required (VERSION 2.8)

project(application-CUDA)
find_package(CUDA)
CUDA_ADD_EXECUTABLE(application-CUDA main.cu helper_cuda.h  helper_functions.h helper_string.h exception.h helper_timer.h helper_image.h)


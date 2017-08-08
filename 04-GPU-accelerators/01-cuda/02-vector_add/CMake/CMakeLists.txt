cmake_minimum_required (VERSION 2.8)

project(application-CUDA)
find_package(CUDA)
CUDA_ADD_EXECUTABLE(application-CUDA vector_add.cu)
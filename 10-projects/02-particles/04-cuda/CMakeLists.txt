cmake_minimum_required (VERSION 2.8)

project(./application-CUDA)
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(CUDA)

CUDA_ADD_EXECUTABLE(./application-CUDA main.cu common.h common.cu)
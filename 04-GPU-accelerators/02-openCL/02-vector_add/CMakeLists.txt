cmake_minimum_required( VERSION 2.6 )
project( application-openCL )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

add_executable( application-openCL main.cpp )
target_link_libraries( application-openCL ${OPENCL_LIBRARIES} )

configure_file(kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel.cl COPYONLY)
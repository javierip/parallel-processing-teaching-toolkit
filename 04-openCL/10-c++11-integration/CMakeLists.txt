cmake_minimum_required( VERSION 2.6 )
project( application-openCL )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

add_executable( application-openCL main.cpp )
target_link_libraries( application-openCL ${OPENCL_LIBRARIES} )

configure_file(vectoradd.cl ${CMAKE_CURRENT_BINARY_DIR}/vectoradd.cl COPYONLY)
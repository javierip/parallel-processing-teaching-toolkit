cmake_minimum_required( VERSION 2.6 )
project( application )

FIND_PACKAGE( MPI REQUIRED)

include_directories( ${MPI_INCLUDE_PATH} )

add_executable( application mpi.cpp common.h common.cpp)

target_link_libraries( application ${MPI_LIBRARIES} )

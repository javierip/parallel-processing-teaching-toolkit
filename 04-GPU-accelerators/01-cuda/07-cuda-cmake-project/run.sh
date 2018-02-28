#!/bin/bash
mkdir build
cd build/
cmake ..
make
./CUDAproject
cd ..

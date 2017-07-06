#!/bin/bash
mkdir build
cd build/
cmake ..
make
screen -m -d  -S Counter ./counter
cd ..

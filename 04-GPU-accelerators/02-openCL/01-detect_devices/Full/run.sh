#!/bin/bash
mkdir build
cd build/
cmake ..
make
./application-openCL 
cd ..

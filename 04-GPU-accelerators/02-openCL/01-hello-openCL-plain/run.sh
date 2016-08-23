#!/bin/bash
gcc -Wall -g main.cpp -o application-openCL -I /usr/local/cuda-7.5/include/ -L /usr/lib64/nvidia -l OpenCL
./application-openCL
#!/bin/bash
nvcc -o binary-cuda vectorAdd.cu
./binary-cuda

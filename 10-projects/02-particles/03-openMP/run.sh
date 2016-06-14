#!/bin/bash
mkdir build
cd build/
cmake ..
make
./application -n 100 -o output-openMP-100-particles.txt
cd ..

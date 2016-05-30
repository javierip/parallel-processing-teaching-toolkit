#!/bin/bash
mkdir build
cd build/
cmake ..
make
./application -n 100 -o output-serial-100-particles.txt
cd ..

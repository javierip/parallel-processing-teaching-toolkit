#!/bin/bash
mkdir build
cd build/
cmake ..
make
./application 
./application 2 5 text another_text
cd ..

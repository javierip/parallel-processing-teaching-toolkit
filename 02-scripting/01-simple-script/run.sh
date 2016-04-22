#!/bin/bash
mkdir build
cd build/
cmake ..
./application 
./application 2 5 text another_text
cd ..

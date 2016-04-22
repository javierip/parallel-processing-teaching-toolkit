#!/bin/bash
mkdir build
cd build/
cmake ..
make
for bash_variable in 1 2 4 8
do
   ./application $bash_variable
done

./application 2 5 text another_text
cd ..

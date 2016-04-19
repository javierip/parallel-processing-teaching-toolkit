#!/bin/bash
echo creating Makefile ..
qmake 
echo comiling ..
make
echo run application
./application

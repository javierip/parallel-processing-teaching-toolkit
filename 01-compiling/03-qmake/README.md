## About this example

This example shows a C program compilation and run using QMake.

## Requirements

You should have a compiler installed. Ubuntu Linux:

```bash
sudo apt-get install qt-sdk
```

## Run

Open a terminal and type:

```bash
sh run.sh
```


## Output

A typical output should look like this one. 

```
creating Makefile ..
comiling ..
gcc -c -pipe -O2 -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I../../../../anaconda2/mkspecs/linux-g++ -I. -I../../../../anaconda2/include/QtCore -I../../../../anaconda2/include/QtGui -I../../../../anaconda2/include -I. -I. -o main.o main.c
g++ -Wl,-O1 -Wl,-rpath,/home/javier/anaconda2/lib -o application main.o    -L/home/javier/anaconda2/lib -lQtGui -L/home/javier/anaconda2/lib -L/usr/X11R6/lib -lQtCore -lpthread 
run application
hola UTN!
```

## Extra Resources

The [oficial documentation](https://www.qt.io) for Qt.

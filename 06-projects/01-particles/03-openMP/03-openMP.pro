TEMPLATE = app
TARGET = 
DEPENDPATH += .
INCLUDEPATH += .
QMAKE_CFLAGS   = -fopenmp
QMAKE_CXXFLAGS = -fopenmp
LIBS += -fopenmp


# Input
HEADERS += common.h
SOURCES += common.cpp openmp.cpp

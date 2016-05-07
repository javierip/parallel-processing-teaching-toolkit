TEMPLATE = app
TARGET = binary
DEPENDPATH += .
INCLUDEPATH += . "/usr/local/cuda-7.5/include/"
LIBS += -lOpenCL
QMAKE_CXXFLAGS += -std=c++11

# Input
SOURCES += main.cpp

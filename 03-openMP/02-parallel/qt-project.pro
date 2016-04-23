
TEMPLATE = app
TARGET = binary
DEPENDPATH += .
INCLUDEPATH += .
QMAKE_CFLAGS   = -fopenmp
QMAKE_CXXFLAGS = -fopenmp
LIBS += -fopenmp

# Input
SOURCES += main.c

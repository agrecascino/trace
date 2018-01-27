TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lGLEW -lglfw -lGL -lgomp -lGLU -lpthread -lembree3 -lglut -lOpenCL

QMAKE_CXXFLAGS = -fopenmp -std=c++14 -ffast-math -O2 -pthread

HEADERS += \
    ctpl.h \
    vectorutil.h \
    structs.h \
    cudapatch.h \
    triangle.h \
    sphere.h \
    scene.h
SOURCES += main.cpp \
    triangle.cpp \
    sphere.cpp \
    scene.cpp \
    structs.cpp

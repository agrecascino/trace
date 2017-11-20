TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

LIBS += -lGLEW -lglfw -lGL -lgomp

QMAKE_CXXFLAGS = -fopenmp -O2 -std=c++14

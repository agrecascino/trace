TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

LIBS += -lGLEW -lglfw -lGL -lgomp -lGLU -lpthread

QMAKE_CXXFLAGS = -fopenmp -std=c++14 -ffast-math -O2 -pthread

HEADERS += \
    ctpl.h

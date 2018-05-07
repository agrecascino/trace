TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lGLEW -lglfw -lGL -lgomp -lGLU -lpthread -lembree3 -lglut -lOpenCL -lportaudio

QMAKE_CXXFLAGS = -fopenmp -std=gnu++14 -ffast-math -O0 -pthread

HEADERS += \
    ctpl.h \
    vectorutil.h \
    structs.h \
    cudapatch.h \
    triangle.h \
    sphere.h \
    scene.h \
    player.h \
    libfont.h \
    qdbmp.h \
    halton.hpp \
    sobol.hpp
SOURCES += qdbmp.c \
    main.cpp \
    triangle.cpp \
    sphere.cpp \
    scene.cpp \
    structs.cpp \
    libfont.cpp \
    halton.cpp \
    sobol.cpp

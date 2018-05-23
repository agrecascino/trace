TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lGLEW -lglfw -lGL -lGLU -lpthread -lembree3 -lglut -lOpenCL -lportaudio -ldrm -lgomp
QMAKE_CXXFLAGS = -std=gnu++14 -ffast-math -O0 -pthread -fopenmp -march=native

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

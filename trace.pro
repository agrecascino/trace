TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -lGLEW -lglfw -lGL -lGLU -lpthread -lembree3 -lglut -lOpenCL -lportaudio -ldrm -lomp -ltbb
QMAKE_CXXFLAGS = -std=gnu++14 -fno-math-errno -funsafe-math-optimizations -fassociative-math -freciprocal-math -ffinite-math-only -fno-signed-zeros -fno-trapping-math -frounding-math -fsingle-precision-constant -ffast-math -Ofast -pthread -fopenmp=libomp -march=native

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
    sobol.hpp \
    obj.h
SOURCES += qdbmp.c \
    main.cpp \
    triangle.cpp \
    sphere.cpp \
    scene.cpp \
    structs.cpp \
    libfont.cpp \
    halton.cpp \
    sobol.cpp

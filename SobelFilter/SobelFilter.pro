QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += linkconfig
PKGCONFIG += opencv4
INCLUDEPATH += /usr/include/opencv4
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

####################
CUDA_DIR = /usr/local/cuda-11.4
CUDA_LIBS = -lcudart -lcuda

INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
#LIBS += $$CUDA_LIBS
#LIBS += -L$$CUDA_DIR/lib64 -lcudart
LIBS += -lcudart -lcuda

# GPU architecture
SYSTEM_TYPE = 64
CUDA_ARCH = sm_72
NVCCOPTIONS = -use_fast_math -O2

# Mandatory flags for stepping through the code
##debug {
#    NVCCOPTIONS += -g -G
#}

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
#CUDA_INC = /usr/local/cuda-11.4/include
CUDA_SOURCES += sobel.cu

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

#cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCCOPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}| sed \"s/^.*: //\"
QMAKE_EXTRA_COMPILERS += cuda ```


SOURCES += \
    functions.cpp \
    main.cpp \
    mainwindow.cpp
    #sobel.cu
HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    sobel.cu

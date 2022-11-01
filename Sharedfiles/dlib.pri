# This pri files describes where Dlib's library located and how the app should be linked with the library
win32 {
    #Specify the part of OpenCV path corresponding to compiler version
    win32-msvc2010: DLIB_COMPILER = vc10
    win32-msvc2012: DLIB_COMPILER = vc11
    win32-msvc2013: DLIB_COMPILER = vc12
    win32-msvc2015: DLIB_COMPILER = vc14
    win32-g++:      DLIB_COMPILER = mingw

    #Specify the part of OpenCV path corresponding to target architecture
    win32:contains(QMAKE_TARGET.arch, x86_64){
        DLIB_ARCHITECTURE = x64
    } else {
        DLIB_ARCHITECTURE = x86
    }
    DLIB_ARCHITECTURE = x86

    DLIB_INSTALL_PATH = C:/Programming/3rdParties/Dlib/install/$${DLIB_COMPILER}/$${DLIB_ARCHITECTURE}
    message($${DLIB_INSTALL_PATH})

    INCLUDEPATH += $${DLIB_INSTALL_PATH}/include

    LIBS += -L$${DLIB_INSTALL_PATH}/lib

    openblasbackend {
        message(OpenBLAS backend selected)
        OPENBLAS_INSTALL_PATH = "C:/Program Files (x86)/OpenBLAS"

        LIBS += -L$${OPENBLAS_INSTALL_PATH}/bin \
                -lopenblas
    }
}

LIBS += -ldlib

linux {
    #if Dlib had been built with OpenBLAS
    #CONFIG += openblasbackend
    openblasbackend {
        message(OpenBLAS backend selected)
        LIBS += -lopenblas
    }

    # if Dlib had been built with CUDA
    cudabackend {
        message(CUDA backend selected)
        LIBS += -L/usr/local/cuda/lib64
        LIBS += -lcudnn \
                -lpthread \
                -llapack \
                -lblas \
                -lcuda \
                -lcudart \
                -lcublas \
                -lcurand \
                -lcusolver \
                -ljpeg \
                -lpng
    }
}

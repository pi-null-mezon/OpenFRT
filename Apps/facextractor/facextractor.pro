TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facelandmarks/facemarkcnn.cpp \
    ../../Src/facelandmarks/facemark.cpp

INCLUDEPATH += ../../Src/facedetector \
               ../../Src/facelandmarks

include($${PWD}/../../Sharedfiles/opencv.pri)
include($${PWD}/../../Sharedfiles/dlib.pri)

unix: {
   target.path = /usr/local/bin
   INSTALLS += target
}

HEADERS +=

INCLUDEPATH += $${PWD}/../../../Kaggle/Shared/dlibimgaugment \
               $${PWD}/../../../Kaggle/Shared/opencvimgaugment \
               $${PWD}/../../../Kaggle/Shared/dlibopencvconverter

DEFINES += CNN_FACE_DETECTOR_INPUT_SIZE=128

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facelandmarks/facemarkcnn.cpp \
    facextractionutils.cpp

INCLUDEPATH += ../../Src/facedetector \
               ../../Src/facelandmarks

include($${PWD}/../../Sharedfiles/opencv.pri)
include($${PWD}/../../Sharedfiles/dlib.pri)

unix: {
   target.path = /usr/local/bin
   INSTALLS += target
}

HEADERS += \
    facextractionutils.h

#DEFINES += CNN_FACE_DETECTOR_INPUT_SIZE=350

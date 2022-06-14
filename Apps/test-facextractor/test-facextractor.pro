TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/faceclassifier/facebestshot.cpp \
    ../../Src/faceclassifier/faceclassifier.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facelandmarks/facemarkcnn.cpp \
    #../../Src/facelandmarks/facemarklitecnn.cpp \
    facextractionutils.cpp

INCLUDEPATH += ../../Src/facedetector \
               ../../Src/facelandmarks \
               ../../Src/faceclassifier

include($${PWD}/../../Sharedfiles/opencv.pri)
include($${PWD}/../../Sharedfiles/dlib.pri)

unix: {
   target.path = /usr/local/bin
   INSTALLS += target
}

HEADERS += \
    facextractionutils.h

#DEFINES += CNN_FACE_DETECTOR_INPUT_SIZE=150

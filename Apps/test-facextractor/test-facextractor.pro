TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/faceclassifier/facebestshot.cpp \
    ../../Src/faceclassifier/glassesdetector.cpp \
    ../../Src/faceclassifier/faceblur.cpp \
    ../../Src/faceclassifier/faceliveness.cpp \
    ../../Src/faceclassifier/faceclassifier.cpp \
    ../../Src/faceclassifier/headposepredictor.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facelandmarks/facemarkcnn.cpp \
    ../../Src/facelandmarks/facemarklitecnn.cpp \
    ../../Src/facelandmarks/facemarkdlib.cpp \
    ../../Src/facelandmarks/facemark.cpp \
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
    ../../Src/faceclassifier/faceblur.h \
    ../../Src/faceclassifier/faceliveness.h \
    ../../Src/faceclassifier/headposepredictor.h \
    facextractionutils.h

#DEFINES += CNN_FACE_DETECTOR_INPUT_SIZE=150


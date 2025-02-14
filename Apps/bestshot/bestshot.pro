TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/faceclassifier/emotionsclassifier.cpp \
    ../../Src/faceclassifier/forwardviewdetector.cpp \
    ../../Src/faceclassifier/smiledetector.cpp \
    ../../Src/faceclassifier/facebestshot.cpp \
    ../../Src/faceclassifier/facenoise.cpp \
    ../../Src/faceclassifier/illumestimator.cpp \
    ../../Src/faceclassifier/obstaclesdetector.cpp \
    ../../Src/faceclassifier/openeyedetector.cpp \
    ../../Src/faceclassifier/glassesdetector.cpp \
    ../../Src/faceclassifier/faceblur.cpp \
    ../../Src/faceclassifier/yawndetector.cpp \
    ../../Src/faceclassifier/faceliveness.cpp \
    ../../Src/faceclassifier/faceclassifier.cpp \
    ../../Src/faceclassifier/headposepredictor.cpp \
    ../../Src/faceclassifier/rotateclassifier.cpp \
    ../../Src/faceclassifier/crfiqaestimator.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facedetector/yunet2023fd.cpp \
    ../../Src/facelandmarks/facemarkcnn.cpp \
    ../../Src/facelandmarks/facemarklitecnn.cpp \
    ../../Src/facelandmarks/facemarkdlib.cpp \
    ../../Src/facelandmarks/facemarkonnx.cpp \
    ../../Src/facelandmarks/facemarkwithpose.cpp \
    ../../Src/facelandmarks/facemark.cpp


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
    ../../Src/faceclassifier/headposepredictor.h

DEFINES += CNN_FACE_DETECTOR_INPUT_SIZE=96


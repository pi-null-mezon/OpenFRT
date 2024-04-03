TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle

SOURCES += main.cpp \
    ../../Src/facedetector/yunetfacedetector.cpp \
    ../../Src/facedetector/cnnfacedetector.cpp \
    ../../Src/facedetector/facedetector.cpp \
    ../../Src/facelandmarks/facemarkwithpose.cpp \
    ../../Src/facelandmarks/facemark.cpp \
    ../../Src/faceclassifier/faceclassifier.cpp

INCLUDEPATH += ../../Src/facedetector \
               ../../Src/facelandmarks \
               ../../Src/faceclassifier

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

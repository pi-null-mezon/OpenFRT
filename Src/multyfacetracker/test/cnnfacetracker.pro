TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

TARGET = cnnfacetracker

SOURCES += \
        main.cpp

DEFINES += APP_NAME=\\\"$${TARGET}\\\"

include($${PWD}/../../../Sharedfiles/opencv.pri)
include($${PWD}/../../../Sharedfiles/dlib.pri)


SOURCES += $${PWD}/../../facedetector/facedetector.cpp \
            $${PWD}/../../facedetector/cnnfacedetector.cpp \
            $${PWD}/../../multyfacetracker/multyfacetracker.cpp \
            $${PWD}/../../facelandmarks/facemarkdlib.cpp

HEADERS += \
    $${PWD}/../../facedetector/facedetector.h \
    $${PWD}/../../facedetector/cnnfacedetector.h \
    $${PWD}/../../multyfacetracker/multyfacetracker.h \
    $${PWD}/../../facelandmarks/facemarkdlib.h

INCLUDEPATH += $${PWD}/../../facedetector \
               $${PWD}/../../multyfacetracker \
               $${PWD}/../../facelandmarks

linux {
    #DEFINES += TRY_TO_USE_CUDA
}

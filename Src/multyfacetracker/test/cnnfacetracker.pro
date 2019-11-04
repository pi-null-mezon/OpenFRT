TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

TARGET = cnnfacetracker

SOURCES += \
        main.cpp

DEFINES += APP_NAME=\\\"$${TARGET}\\\"

include($${PWD}/../../../Sharedfiles/opencv.pri)

DEFINES += TRY_TO_USE_CUDA

SOURCES += $${PWD}/../../facedetector/facedetector.cpp \
            $${PWD}/../../facedetector/cnnfacedetector.cpp \
            $${PWD}/../../multyfacetracker/multyfacetracker.cpp

HEADERS += \
    $${PWD}/../../facedetector/facedetector.h \
    $${PWD}/../../facedetector/cnnfacedetector.h \
    $${PWD}/../../multyfacetracker/multyfacetracker.h

INCLUDEPATH += $${PWD}/../../facedetector \
                $${PWD}/../../multyfacetracker

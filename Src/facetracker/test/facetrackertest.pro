QT += core
QT -= gui

CONFIG += c++11

TARGET = facetracker
VERSION = 1.0.2.0

CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

OFRT_PATH = C:/Programming/ofrt

SOURCES += main.cpp \
           $${PWD}/../../../Sources/facetracker/facetracker.cpp

HEADERS += $${PWD}/../../../Sources/facetracker/facetracker.h

INCLUDEPATH += $${PWD}/../../../Sources/facetracker

include( $${PWD}/../../../Sharedfiles/opencv.pri )
include( $${PWD}/../../../Sharedfiles/opencl.pri )
include( $${PWD}/../../../Sharedfiles/openmp.pri )
include( $${PWD}/../../../Sharedfiles/dlib.pri )

#CONFIG += designbuild

designbuild {
    DEFINES += OPENCV_DIRECTORY=\\\"$${OPENCV_DIR}/..\\\"
    message(Design build mode enabled)
}

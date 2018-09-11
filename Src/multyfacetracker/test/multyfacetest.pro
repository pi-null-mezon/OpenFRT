QT += core
QT -= gui

CONFIG += c++11

TARGET = multyfacetracker
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

OFRT_PATH = C:/Programming/ofrt

SOURCES += main.cpp \
           $${OFRT_PATH}/Sources/facetracker/facetracker.cpp \
           $${OFRT_PATH}/Sources/multyfacetracker/multyfacetracker.cpp

HEADERS += $${OFRT_PATH}/Sources/multyfacetracker/multyfacetracker.h

INCLUDEPATH += $${OFRT_PATH}/Sources/multyfacetracker \
               $${OFRT_PATH}/Sources/facetracker

include( $${OFRT_PATH}/Sharedfiles/opencv.pri )
include( $${OFRT_PATH}/Sharedfiles/opencl.pri)
include( $${OFRT_PATH}/Sharedfiles/openmp.pri)

CONFIG += designbuild

designbuild {
    DEFINES += OPENCV_DIRECTORY=\\\"$${OPENCV_DIR}/..\\\"
    message(Design build mode enabled)
}

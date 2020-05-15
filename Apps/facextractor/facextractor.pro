TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle

SOURCES += main.cpp

include($${PWD}/../../Sharedfiles/opencv.pri)
include($${PWD}/../../Src/multyfacetracker/facetracker.pri)

unix: {
   target.path = /usr/local/bin
   INSTALLS += target
}

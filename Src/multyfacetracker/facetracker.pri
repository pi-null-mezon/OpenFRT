SOURCES += $${PWD}/multyfacetracker.cpp \
        $${PWD}/qmultyfacetracker.cpp \
        $${PWD}/../facedetector/facedetector.cpp \
        $${PWD}/../facedetector/cnnfacedetector.cpp
					
HEADERS += $${PWD}/multyfacetracker.h \
        $${PWD}/qmultyfacetracker.h \
        $${PWD}/../facedetector/facedetector.h \
        $${PWD}/../facedetector/cnnfacedetector.h

INCLUDEPATH += $${PWD} \
            $${PWD}/../facedetector

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "facedetector.h"
#include "cnnfacedetector.h"
#include "multifacetracker.h"

using namespace std;

const cv::String keys = "{videodev v | 0 | number of videodevice to open}"
                        "{dscr       |   | network description file name}"
                        "{model      |   | network weights file name}";

int main(int argc, char **argv)
{
    cv::CommandLineParser cmdparser(argc,argv,keys);
    cmdparser.about("Test application");
    if(argc == 1) {
        cmdparser.printMessage();
        return 0;
    }
    if(!cmdparser.has("dscr")) {
        cout << "No description file provided! Abort..." << endl;
        return 1;
    }
    if(!cmdparser.has("model")) {
        cout << "No model file provided! Abort..." << endl;
        return 2;
    }

    cv::VideoCapture videocapture;
    if(!videocapture.open(cmdparser.get<int>("videodev"))) {
        cout << "Can not open videodevice with id " << cmdparser.get<int>("videodev") <<endl;
        return 3;
    }

    auto dPtr = cv::ofrt::CNNFaceDetector::createDetector(cmdparser.get<string>("dscr"),cmdparser.get<string>("model"));
    cv::ofrt::MultiFaceTracker mfacetracker(dPtr,4);

    cv::Mat framemat;
    double _frametimems, _timemark = cv::getTickCount();
    std::string timestr;
    while(videocapture.read(framemat)) {       
        // Frame processing block
        /*std::vector<cv::Rect> _faces = dPtr->detectFaces(framemat);
        for(size_t i = 0; i < _faces.size(); ++i) {
            cv::rectangle(framemat,_faces[i],cv::Scalar(0,255,127),1,cv::LINE_AA);
        }*/

        auto _vfaces = mfacetracker.getResizedFaceImages(framemat,cv::Size(150,150));
        auto _vtrackedfaces = mfacetracker.getTrackedFaces();
        for(size_t i = 0; i < _vtrackedfaces.size(); ++i) {
            if(_vtrackedfaces[i].getFramesTracked() > 0) {
                string label = std::to_string(_vtrackedfaces[i].getUuid());
                cv::Rect _rect = _vtrackedfaces[i].getRect(4);
                cv::rectangle(framemat,_rect,cv::Scalar(0,255,127),1,cv::LINE_AA);
                cv::putText(framemat,label,_rect.tl() - cv::Point(2,2),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1,cv::LINE_AA);
            }
        }
        /*for(size_t i = 0; i < _vfaces.size(); ++i) {
            cv::imshow(std::to_string(i),_vfaces[i]);
        }*/


        // Performance measurements
        _frametimems = 1000.0 * (cv::getTickCount() - _timemark) / cv::getTickFrequency();
        _timemark = cv::getTickCount();
        timestr = cv::format("%.2f ms; Press 'esc' to exit, and 's' to set videodev settings", _frametimems);
        cv::putText(framemat,timestr,cv::Point(15,framemat.rows - 15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),1,cv::LINE_AA);
        cv::putText(framemat,timestr,cv::Point(14,framemat.rows - 16),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
        cv::imshow(APP_NAME,framemat);
        char key = cv::waitKey(1);
        if(key == 27)
            break;
        if(key == 's')
            videocapture.set(CV_CAP_PROP_SETTINGS,0.0);
    }
    return 0;
}

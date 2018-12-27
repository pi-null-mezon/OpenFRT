#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "facedetector.h"
#include "cnnfacedetector.h"
#include "multyfacetracker.h"

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
    } else {
        /*videocapture.set(cv::CAP_PROP_FRAME_WIDTH,1280);
        videocapture.set(cv::CAP_PROP_FRAME_HEIGHT,720);*/
    }

    auto dPtr = cv::ofrt::CNNFaceDetector::createDetector(cmdparser.get<string>("dscr"),cmdparser.get<string>("model"));
    dPtr->setPortions(1.4,1.2);
    cv::ofrt::MultyFaceTracker mfacetracker(dPtr,16);


    cv::Mat framemat;
    double _frametimems, _timemark = cv::getTickCount();
    std::string timestr;
    while(videocapture.read(framemat)) {       
        // Frame processing block
        auto _vfaces = mfacetracker.getResizedFaceImages(framemat,cv::Size(170,226),2);
        for(size_t i = 0; i < _vfaces.size(); ++i) {
            cv::ofrt::TrackedFace *_tf = mfacetracker.at(_vfaces[i].first);
            if(_tf->getFramesTracked() > 0) {
                string label = string("FT# ") + to_string(_vfaces[i].first) + string(", face guid: ") + std::to_string(_tf->getUuid());
                cv::Rect _rect = _tf->getRect(2);
                cv::rectangle(framemat,_rect,cv::Scalar(0,255,127),1,cv::LINE_AA);
                cv::putText(framemat,label,_rect.tl() - cv::Point(0,10),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1,cv::LINE_AA);
            }
        }
        for(size_t i = 0; i < _vfaces.size(); ++i) {
            cv::imshow(std::to_string(i),_vfaces[i].second);
        }
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

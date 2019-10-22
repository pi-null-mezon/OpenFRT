#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>

#include "facedetector.h"
#include "cnnfacedetector.h"
#include "multyfacetracker.h"

using namespace std;

const cv::String keys = "{videodev v  | 0 | number of videodevice to open}"
                        "{dscr d      |   | facedetection model description file name}"
                        "{model m     |   | facedetection model weights file name}"
                        "{landmarks l |   | facial landmarks model (optional)}";

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

    cv::Ptr<cv::face::Facemark> flandmarks = cv::face::createFacemarkLBF();
    bool performlandmarksdetection = false;
    if(cmdparser.has("landmarks")) {
        flandmarks->loadModel(cmdparser.get<std::string>("landmarks"));
        performlandmarksdetection = true;
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
    dPtr->setPortions(1,0.7);
    cv::ofrt::MultyFaceTracker mfacetracker(dPtr,16);


    cv::Mat framemat, mattoshow;
    double _frametimems, _timemark = cv::getTickCount();
    std::string timestr;
    while(videocapture.read(framemat)) {
        mattoshow = framemat.clone();
        // Frame processing block
        auto _vfaces = mfacetracker.getResizedFaceImages(framemat,cv::Size(226,226),2);
        std::vector<cv::Rect> _vrects;
        _vrects.reserve(mfacetracker.maxFaces());
        for(size_t i = 0; i < _vfaces.size(); ++i) {
            cv::ofrt::TrackedFace *_tf = mfacetracker.at(_vfaces[i].first);
            if(_tf->getFramesTracked() > 0) {
                string label = string("FT# ") + to_string(_vfaces[i].first) + string(", face guid: ") + std::to_string(_tf->getUuid());
                cv::Rect _rect = _tf->getRect(2);
                if(performlandmarksdetection)
                    _vrects.push_back(_rect);
                cv::rectangle(mattoshow,_rect,cv::Scalar(0,255,127),1,cv::LINE_AA);
                cv::putText(mattoshow,label,_rect.tl() - cv::Point(0,10),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1,cv::LINE_AA);
            }
        }
        if(performlandmarksdetection && (_vrects.size() > 0)) {
            std::vector<std::vector<cv::Point2f>> _vlandmarks;
            flandmarks->fit(framemat,_vrects,_vlandmarks);
            for(size_t i = 0; i < _vlandmarks.size(); ++i)
                for(size_t j = 0; j < _vlandmarks[i].size(); ++j)
                    cv::circle(mattoshow,_vlandmarks[i][j],1,cv::Scalar(255,255,255),1,cv::LINE_AA);
        }
        for(size_t i = 0; i < _vfaces.size(); ++i) {
            cv::imshow(std::to_string(i),_vfaces[i].second);
        }
        // Performance measurements
        _frametimems = 1000.0 * (cv::getTickCount() - _timemark) / cv::getTickFrequency();
        _timemark = cv::getTickCount();
        timestr = cv::format("%.2f ms; Press 'esc' to exit, and 's' to set videodev settings", _frametimems);
        cv::putText(mattoshow,timestr,cv::Point(15,framemat.rows - 15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),1,cv::LINE_AA);
        cv::putText(mattoshow,timestr,cv::Point(14,framemat.rows - 16),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
        cv::imshow(APP_NAME,mattoshow);
        char key = cv::waitKey(1);
        if(key == 27)
            break;
        if(key == 's')
            videocapture.set(cv::CAP_PROP_SETTINGS,0.0);
    }
    return 0;
}

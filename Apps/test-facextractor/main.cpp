#include <QStringList>
#include <QUuid>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "cnnfacedetector.h"
#include "yunetfacedetector.h"
#include "facemarkcnn.h"
#include "facemarklitecnn.h"

#include "facextractionutils.h"

const cv::String _options = "{help h               |                        | this help                                                     }"
                            "{device               | 0                      | video device                                   }"
                            "{videofile            |                        | input videofile, if used will be processed instead of inputdir}"
                            "{facedetmodel m       | res10_300x300_ssd_iter_140000_fp16.caffemodel | face detector model                    }"
                            "{facedetdscr d        | deploy_lowres.prototxt | face detector description                                     }"
                            "{confthresh           | 0.8                    | confidence threshold for the face detector                    }"
                            "{facelandmarksmodel l | facelandmarks_net.dat  | face landmarks model (68 points)                              }"
                            "{targeteyesdistance   | 90.0                   | target distance between eyes                                  }"
                            "{targetwidth          | 300                    | target image width                                            }"
                            "{targetheight         | 400                    | target image height                                           }"
                            "{h2wshift             | 0                      | additional horizontal shift to face crop in portion of target width}"
                            "{v2hshift             | 0                      | additional vertical shift to face crop in portion of target height }"
                            "{rotate               | true                   | apply rotation to make eyes-line horizontal aligned           }"
                            "{visualize v          | true                   | enable/disable visualization option                           }";

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif
    cv::CommandLineParser _cmdparser(argc,argv,_options);
    _cmdparser.about("Utility to test face detection and alignment speed");
    if(_cmdparser.has("help") || argc == 1) {
        _cmdparser.printMessage();
        return 0;
    }


    if(!_cmdparser.has("facedetmodel")) {
        qWarning("You have not specified face detector model filename! Abort...");
        return 3;
    }
    if(!_cmdparser.has("facedetdscr")) {
        qWarning("You have not specified face detector description filename! Abort...");
        return 4;
    }
    /*cv::Ptr<cv::ofrt::FaceDetector> facedetector = cv::ofrt::CNNFaceDetector::createDetector(_cmdparser.get<std::string>("facedetdscr"),
                                                                                             _cmdparser.get<std::string>("facedetmodel"),
                                                                                             _cmdparser.get<float>("confthresh"));*/
    cv::Ptr<cv::ofrt::FaceDetector> facedetector = cv::ofrt::YuNetFaceDetector::createDetector(_cmdparser.get<std::string>("facedetmodel"),
                                                                                             _cmdparser.get<float>("confthresh"));
    cv::Ptr<cv::face::Facemark> facelandmarker = cv::face::createFacemarkCNN();
    facelandmarker->loadModel(_cmdparser.get<std::string>("facelandmarksmodel"));


    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    float _targeteyesdistance = _cmdparser.get<float>("targeteyesdistance");
    const bool _visualize = _cmdparser.get<bool>("visualize");
    const float h2wshift = _cmdparser.get<float>("h2wshift");
    const float v2hshift = _cmdparser.get<float>("v2hshift");
    const bool rotate = _cmdparser.get<bool>("rotate");

    cv::VideoCapture videocapture;
    if(_cmdparser.has("videofile")) {
        std::string filename = _cmdparser.get<std::string>("videofile");
        if(videocapture.open(filename)) {
            qInfo("Video file has been opened successfully");
            cv::Mat frame;
            unsigned long framenum = 0;
            while(videocapture.read(frame)) {
                float t0 = cv::getTickCount();
                const std::vector<std::vector<cv::Point2f>> _faces = detectFacesLandmarks(frame,facedetector,facelandmarker);
                /*cv::Ptr<cv::ofrt::YuNetFaceDetector> yunfd = facedetector.dynamicCast<cv::ofrt::YuNetFaceDetector>();
                const std::vector<std::vector<cv::Point2f>> _faces = yunfd->detectLandmarks(frame);*/
                float duration_ms = 1000.0f * (cv::getTickCount() - t0) / cv::getTickFrequency();
                if(_faces.size() != 0) {
                    //qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
                    for(size_t j = 0; j < _faces.size(); ++j) {
                        const cv::Mat _facepatch = extractFacePatch(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                        for(const auto &pt: _faces[j])
                            cv::circle(frame,pt,1,cv::Scalar(0,255,0),-1,cv::LINE_AA);
                    }
                }
                if(_visualize) {
                    cv::putText(frame,QString("%1 ms").arg(QString::number(duration_ms,'f',1)).toStdString(),
                                cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(frame,QString("%1 ms").arg(QString::number(duration_ms,'f',1)).toStdString(),
                                cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

                    cv::imshow("Probe",frame);
                    cv::waitKey(1);
                }
                framenum++;
            }
        } else {
            qInfo("Can not open '%s'! Abort...", filename.c_str());
            return 5;
        }
    } else if(_cmdparser.has("device")) {
        int device = _cmdparser.get<int>("device");
        if(videocapture.open(device)) {
            videocapture.set(cv::CAP_PROP_FRAME_WIDTH,640);
            videocapture.set(cv::CAP_PROP_FRAME_HEIGHT,480);
            videocapture.set(cv::CAP_PROP_FPS,30);
            qInfo("Video device %d has been opened successfully", device);
            cv::Mat frame;
            unsigned long framenum = 0;
            while(videocapture.read(frame)) {
                float t0 = cv::getTickCount();
                const std::vector<std::vector<cv::Point2f>> _faces = detectFacesLandmarks(frame,facedetector,facelandmarker);
                /*cv::Ptr<cv::ofrt::YuNetFaceDetector> yunfd = facedetector.dynamicCast<cv::ofrt::YuNetFaceDetector>();
                const std::vector<std::vector<cv::Point2f>> _faces = yunfd->detectLandmarks(frame);*/
                float duration_ms = 1000.0f * (cv::getTickCount() - t0) / cv::getTickFrequency();
                if(_faces.size() != 0) {
                    //qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
                    for(size_t j = 0; j < _faces.size(); ++j) {
                        const cv::Mat _facepatch = extractFacePatch(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                        for(const auto &pt: _faces[j])
                            cv::circle(frame,pt,1,cv::Scalar(0,255,0),-1,cv::LINE_AA);
                    }                  
                }
                if(_visualize) {
                    cv::putText(frame,QString("%1 ms").arg(QString::number(duration_ms,'f',1)).toStdString(),
                                cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(frame,QString("%1 ms").arg(QString::number(duration_ms,'f',1)).toStdString(),
                                cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

                    cv::imshow("Probe",frame);
                    cv::waitKey(1);
                }
                framenum++;
            }
        } else {
            qInfo("Can not open device %d! Abort...", device);
            return 5;
        }
    } else {
        qWarning("You have not specified any input. Nor videofile nor directory! Abort...");
        return 1;
    }
    return 0;
}


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
#include "facebestshot.h"
#include "faceblur.h"
#include "headposepredictor.h"


#include "facextractionutils.h"

const cv::String _options = "{help h               |                        | this help                                                     }"
                            "{device               | 0                      | video device                                                  }"
                            "{videofile            |                        | input videofile, if used will be processed instead of inputdir}"
                            "{facedetmodel m       | res10_300x300_ssd_iter_140000_fp16.caffemodel | face detector model                    }"
                            "{facedetdscr d        | deploy_lowres.prototxt | face detector description                                     }"
                            "{confthresh           | 0.5                    | confidence threshold for the face detector                    }"
                            "{facelandmarksmodel l | facelandmarks_net.dat  | face landmarks model (68 points)                              }"
                            "{targeteyesdistance   | 60.0                   | target distance between eyes                                  }"
                            "{targetwidth          | 200                    | target image width                                            }"
                            "{targetheight         | 300                    | target image height                                           }"
                            "{h2wshift             | 0                      | additional horizontal shift to face crop in portion of target width}"
                            "{v2hshift             | 0                      | additional vertical shift to face crop in portion of target height }"
                            "{rotate               | true                   | apply rotation to make eyes-line horizontal aligned           }"
                            "{headposemodel        | headpose_net_lite.dat  | head pose predictor                                           }"
                            "{blurmodel            | blur_net_lite.dat      | face blureness detector                                       }"
                            "{maxblur              | 0.99                   | max blur allowed                                              }";

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
    cv::Ptr<cv::face::Facemark> facelandmarker = cv::face::createFacemarkLiteCNN();
    facelandmarker->loadModel(_cmdparser.get<std::string>("facelandmarksmodel"));

    cv::Ptr<cv::ofrt::FaceClassifier> blurenessdetector = cv::ofrt::FaceBlur::createClassifier(_cmdparser.get<std::string>("blurmodel"));
    cv::Ptr<cv::ofrt::FaceClassifier> headposepredictor = cv::ofrt::HeadPosePredictor::createClassifier(_cmdparser.get<std::string>("headposemodel"));

    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    float _targeteyesdistance = _cmdparser.get<float>("targeteyesdistance");
    const float h2wshift = _cmdparser.get<float>("h2wshift");
    const float v2hshift = _cmdparser.get<float>("v2hshift");
    const bool rotate = _cmdparser.get<bool>("rotate");
    const float max_blur = _cmdparser.get<float>("maxblur");

    cv::VideoCapture videocapture;
    if(_cmdparser.has("videofile")) {
        std::string filename = _cmdparser.get<std::string>("videofile");
        if(videocapture.open(filename)) {
            qInfo("Video file has been opened successfully");          
        } else {
            qInfo("Can not open ! Abort...");
            return 5;
        }
    } else if(_cmdparser.has("device")) {
        int device = _cmdparser.get<int>("device");
        if(videocapture.open(device)) {
            videocapture.set(cv::CAP_PROP_FRAME_WIDTH,640);
            videocapture.set(cv::CAP_PROP_FRAME_HEIGHT,480);
            videocapture.set(cv::CAP_PROP_FPS,30);
            qInfo("Video device %d has been opened successfully", device);         
        } else {
            qInfo("Can not open device %d! Abort...", device);
            return 5;
        }
    } else {
        qWarning("You have not specified any input. Nor videofile nor directory! Abort...");
        return 1;
    }

    cv::Mat frame;
    unsigned long framenum = 0;
    while(videocapture.read(frame)) {
        double t0 = cv::getTickCount();
        const std::vector<std::vector<cv::Point2f>> _faces = detectFacesLandmarks(frame,facedetector,facelandmarker);
        /*cv::Ptr<cv::ofrt::YuNetFaceDetector> yunfd = facedetector.dynamicCast<cv::ofrt::YuNetFaceDetector>();
        const std::vector<std::vector<cv::Point2f>> _faces = yunfd->detectLandmarks(frame);*/
        double duration_ms = 1000.0 * (cv::getTickCount() - t0) / cv::getTickFrequency();
        if(_faces.size() != 0) {
            //qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
            for(size_t j = 0; j < _faces.size(); ++j) {
                t0 = cv::getTickCount();
                float blureness = blurenessdetector->confidence(frame,_faces[j]);
                std::vector<float> angles = headposepredictor->classify(frame,_faces[j]);

                duration_ms += 1000.0f * (cv::getTickCount() - t0) / cv::getTickFrequency();
                if(blureness < max_blur) {
                    cv::Mat _facepatch = extractFacePatch(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                    std::string info = QString("blureness: %1").arg(QString::number(blureness,'f',2)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

                    info = QString("yaw:   %1").arg(QString::number(angles[0],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,39), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
                    info = QString("pitch: %1").arg(QString::number(angles[1],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,60), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,59), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
                    info = QString("roll:  %1").arg(QString::number(angles[2],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,80), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,79), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

                    cv::imshow(std::string("bestshot_") + std::to_string(j), _facepatch);
                }
                for(const auto &pt: _faces[j])
                    cv::circle(frame,pt,1,cv::Scalar(0,255,0),-1,cv::LINE_AA);
            }
        }

        const std::string info = QString("frame processing time: %1 ms").arg(QString::number(duration_ms,'f',1)).toStdString();
        cv::putText(frame,info,cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
        cv::putText(frame,info,cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

        cv::imshow("Probe",frame);
        if(cv::waitKey(1) == 27)
            break;
        framenum++;
    }

    return 0;
}


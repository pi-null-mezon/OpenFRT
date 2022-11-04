#include <QStringList>
#include <QUuid>
#include <QDir>

#include <thread>
#include <mutex>
#include <condition_variable>

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
#include "faceliveness.h"

#include "facextractionutils.h"

const cv::String _options = "{help h               |                        | this help                                                     }"
                            "{device               | 0                      | video device                                                  }"
                            "{frame_width          | 640                    | self explained                                                }"
                            "{frame_height         | 480                    | self explained                                                }"
                            "{frame_rate           | 30                     | self explained                                                }"
                            "{videofile            |                        | input videofile, if used will be processed instead of inputdir}"
                            "{facedetmodel m       | res10_300x300_ssd_iter_140000_fp16.caffemodel | face detector model                    }"
                            "{facedetdscr d        | deploy_lowres.prototxt | face detector description                                     }"
                            "{confthresh           | 0.5                    | confidence threshold for the face detector                    }"
                            "{facelandmarksmodel l | facelandmarks_net.dat  | face landmarks model (68 points)                              }"
                            "{targeteyesdistance   | 60.0                   | target distance between eyes                                  }"
                            "{targetwidth          | 250                    | target image width                                            }"
                            "{targetheight         | 300                    | target image height                                           }"
                            "{h2wshift             | 0                      | additional horizontal shift to face crop in portion of target width}"
                            "{v2hshift             | 0                      | additional vertical shift to face crop in portion of target height }"
                            "{rotate               | true                   | apply rotation to make eyes-line horizontal aligned           }"
                            "{headposemodel        | headpose_net_lite.dat  | head pose predictor                                           }"
                            "{maxangle             | 90.0                   | max angle allowed (any of pitch, yaw, roll)                   }"
                            "{blurmodel            | blur_net_lite.dat      | face blureness detector                                       }"
                            "{maxblur              | 1.0                    | max blur allowed                                              }"
                            "{fast                 | true                   | make single inference for each detector and classifier        }"
                            "{multithreaded        | true                   | process tasks in parallel threads when possible               }"
                            "{livenessmodel        | liveness_net_lite_v2.dat  | face liveness detector                                        }";

std::vector<float> frame_times(15,0.0f);
size_t frame_times_pos = 0;

float average(const std::vector<float> &values) {
    if(values.size() == 0)
        return 0.0f;
    if(values.size() == 1)
        return values[0];
    return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif
    dlib::set_dnn_prefer_fastest_algorithms();

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
    /*cv::Ptr<cv::ofrt::FaceDetector> facedetector1 = cv::ofrt::CNNFaceDetector::createDetector(_cmdparser.get<std::string>("facedetdscr"),
                                                                                             _cmdparser.get<std::string>("facedetmodel"),
                                                                                             _cmdparser.get<float>("confthresh"));*/
    cv::Ptr<cv::ofrt::FaceDetector> facedetector = cv::ofrt::YuNetFaceDetector::createDetector(_cmdparser.get<std::string>("facedetmodel"),
                                                                                             _cmdparser.get<float>("confthresh"));
    cv::Ptr<cv::ofrt::Facemark> facelandmarker = cv::ofrt::FacemarkLiteCNN::create(_cmdparser.get<std::string>("facelandmarksmodel"));

    cv::Ptr<cv::ofrt::FaceClassifier> blurenessdetector = cv::ofrt::FaceBlur::createClassifier(_cmdparser.get<std::string>("blurmodel"));
    cv::Ptr<cv::ofrt::FaceClassifier> headposepredictor = cv::ofrt::HeadPosePredictor::createClassifier(_cmdparser.get<std::string>("headposemodel"));
    cv::Ptr<cv::ofrt::FaceClassifier> livenessdetector = cv::ofrt::FaceLiveness::createClassifier(_cmdparser.get<std::string>("livenessmodel"));

    qInfo("Configuration:");
    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    float _targeteyesdistance = _cmdparser.get<float>("targeteyesdistance");
    const float h2wshift = _cmdparser.get<float>("h2wshift");
    const float v2hshift = _cmdparser.get<float>("v2hshift");
    const bool rotate = _cmdparser.get<bool>("rotate");
    const float max_blur = _cmdparser.get<float>("maxblur");
    const float max_angle = _cmdparser.get<float>("maxangle");
    const bool fast = _cmdparser.get<bool>("fast");
    qInfo(" - fast prediction - %s", fast ? "ON" : "OFF");
    const bool multithreaded = _cmdparser.get<bool>("multithreaded");
    qInfo(" - multithreaded - %s", multithreaded ? "ON" : "OFF");

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
            videocapture.set(cv::CAP_PROP_FRAME_WIDTH,_cmdparser.get<int>("frame_width"));
            videocapture.set(cv::CAP_PROP_FRAME_HEIGHT,_cmdparser.get<int>("frame_height"));
            videocapture.set(cv::CAP_PROP_FPS,_cmdparser.get<int>("frame_rate"));
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

    bool finish = false;
    std::mutex mtx;
    std::condition_variable cnd;
    std::vector<cv::Point2f> landmarks;
    bool blureness_ready;
    float blureness = 0.0f;
    bool liveness_ready;
    float liveness_score = 0.0f;
    std::vector<float> angles;

    std::thread *blureness_thread = nullptr;
    std::thread *liveness_thread = nullptr;
    if(multithreaded) {
        blureness_thread = new std::thread([&](){
            while(true) {
                std::unique_lock<std::mutex> lck(mtx);
                cnd.wait(lck, [&](){return ((!blureness_ready && landmarks.size() != 0) || finish);});
                if(finish)
                    break;
                blureness = blurenessdetector->process(frame,landmarks,fast)[0];
                blureness_ready = true;
                cnd.notify_all();
            }
        });
        liveness_thread = new std::thread([&](){
            while(true) {
                std::unique_lock<std::mutex> lck(mtx);
                cnd.wait(lck, [&](){return ((!liveness_ready && landmarks.size() != 0) || finish);});
                if(finish)
                    break;
                liveness_score = livenessdetector->process(frame,landmarks,fast)[0];
                liveness_ready = true;
                cnd.notify_all();
            }
        });
    }

    size_t faces_found_erlier = 0;
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
                if(multithreaded) {
                    {
                        std::lock_guard<std::mutex> lck(mtx);
                        blureness_ready = false;
                        liveness_ready = false;
                        landmarks = _faces[j];
                        cnd.notify_all();
                    }
                    angles = headposepredictor->process(frame,landmarks,fast);
                    {
                        std::unique_lock<std::mutex> lck(mtx);
                        cnd.wait(lck,[&](){return (blureness_ready);});
                        cnd.wait(lck,[&](){return (liveness_ready);});
                    }
                } else {
                    landmarks = _faces[j];
                    blureness = blurenessdetector->process(frame,landmarks,fast)[0];
                    liveness_score = livenessdetector->process(frame,landmarks,fast)[0];
                    angles = headposepredictor->process(frame,landmarks,fast);
                }
                duration_ms += 1000.0f * (cv::getTickCount() - t0) / cv::getTickFrequency();
                frame_times[frame_times_pos++] = duration_ms;
                if (frame_times_pos == frame_times.size())
                    frame_times_pos = 0;
                if((blureness < max_blur) && (std::abs(*std::max_element(angles.begin(),angles.end())) < max_angle)) {
                    cv::Mat _facepatch = extractFacePatch(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                    std::string info = QString("blureness: %1").arg(QString::number(blureness,'f',2)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,blureness < 0.5 ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255),1,cv::LINE_AA);

                    info = QString("yaw:   %1").arg(QString::number(angles[0],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,39), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
                    info = QString("pitch: %1").arg(QString::number(angles[1],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,60), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,59), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);
                    info = QString("roll:  %1").arg(QString::number(angles[2],'f',0)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,80), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,79), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

                    info = QString("liveness:  %1").arg(QString::number(liveness_score,'f',2)).toStdString();
                    cv::putText(_facepatch,info,cv::Point(20,100), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
                    cv::putText(_facepatch,info,cv::Point(19,99), cv::FONT_HERSHEY_SIMPLEX,0.5,liveness_score > 0.5 ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255),1,cv::LINE_AA);

                    cv::imshow(std::string("bestshot_") + std::to_string(j), _facepatch);
                }
                for(const auto &pt: _faces[j])
                    cv::circle(frame,pt,1,cv::Scalar(0,255,0),-1,cv::LINE_AA);
            }
        }

        const std::string info = QString("frame processing time: %1 ms").arg(QString::number(average(frame_times),'f',1)).toStdString();
        cv::putText(frame,info,cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1,cv::LINE_AA);
        cv::putText(frame,info,cv::Point(19,19), cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255),1,cv::LINE_AA);

        if(_faces.size() < faces_found_erlier){
            for(size_t j = _faces.size(); j < faces_found_erlier; ++j)
                cv::destroyWindow(std::string("bestshot_") + std::to_string(j));
        }
        faces_found_erlier = _faces.size();

        cv::imshow("Probe",frame);
        if(cv::waitKey(1) == 27) {
            {
                std::lock_guard<std::mutex> lck(mtx);
                finish = true;
                cnd.notify_all();
            }
            break;
        }
        framenum++;
    }

    if(multithreaded) {
        blureness_thread->join();
        liveness_thread->join();
        delete blureness_thread;
        delete liveness_thread;
    }
    return 0;
}


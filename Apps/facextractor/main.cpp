#include <QStringList>
#include <QUuid>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "cnnfacedetector.h"
#include "yunetfacedetector.h"
#include "facemarkcnn.h"
#include "facemarkwithpose.h"
#include "faceclassifier.h"

#include "dlibimgaugment.h"
#include "opencvimgaugment.h"

const cv::String _options = "{help h               |                        | this help                                                     }"
                            "{inputdir i           |                        | input directory with images                                   }"
                            "{videofile            |                        | input videofile, if used will be processed instead of inputdir}"
                            "{videodir v           |                        | input directory with videos                                   }"
                            "{outputdir o          |                        | output directory with images                                  }"
                            "{facedetmodel m       | ./res10_300x300_ssd_iter_140000_fp16.caffemodel | face detector model                  }"
                            "{facedetdscr d        | ./deploy_lowres.prototxt | face detector description                                   }"
                            "{confthresh           | 0.5                    | confidence threshold for the face detector                    }"
                            "{facelandmarksmodel l | ./facelandmarks_net.dat  | face landmarks model (68 points)                            }"
                            "{targeteyesdistance   | 30.0                   | target distance between eyes                                  }"
                            "{targetwidth          | 150                    | target image width                                            }"
                            "{targetheight         | 150                    | target image height                                           }"
                            "{h2wshift             | 0                      | additional horizontal shift to face crop in portion of target width}"
                            "{v2hshift             | 0                      | additional vertical shift to face crop in portion of target height }"
                            "{rotate               | true                   | apply rotation to make eyes-line horizontal aligned           }"
                            "{videostrobe          | 5                      | only each videostrobe frame will be processed                 }"
                            "{visualize v          | false                  | enable/disable visualization option                           }"
                            "{preservefilenames    | false                  | enable/disable filenames preservation                         }"
                            "{preservesubdirs      | false                  | enable/disable subdirs preservation                           }"
                            "{codec                | png                    | encoding format to save extracted faces                       }";

std::vector<QString> find_all_subdirs_in_path(const QString &path);

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif
    cv::CommandLineParser _cmdparser(argc,argv,_options);
    _cmdparser.about("Utility to crop faces from images or videos with roll alignment");
    if(_cmdparser.has("help") || argc == 1) {
        _cmdparser.printMessage();
        return 0;
    }
    if(!_cmdparser.has("outputdir")) {
        qWarning("You have not specified output directory! Abort...");
        return 2;
    }
    const cv::String outputdirname = _cmdparser.get<cv::String>("outputdir");
    QDir _outdir(outputdirname.c_str());
    if(!_outdir.exists())
        _outdir.mkpath(_outdir.absolutePath());

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
    cv::Ptr<cv::ofrt::Facemark> facelandmarker = cv::ofrt::FacemarkWithPose::create(_cmdparser.get<std::string>("facelandmarksmodel"));

    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    float _targeteyesdistance = _cmdparser.get<float>("targeteyesdistance");
    const bool _visualize = _cmdparser.get<bool>("visualize");
    const bool _preservefilenames = _cmdparser.get<bool>("preservefilenames");
    const bool _preservesubdirnames = _cmdparser.get<bool>("preservesubdirs");
    const float h2wshift = _cmdparser.get<float>("h2wshift");
    const float v2hshift = _cmdparser.get<float>("v2hshift");
    const bool rotate = _cmdparser.get<bool>("rotate");
    const QString extension = _cmdparser.get<std::string>("codec").c_str();
    const unsigned int strobe = _cmdparser.get<unsigned int>("videostrobe");

    cv::RNG cvrng(7);

    if(_cmdparser.has("videodir")) {
        size_t _facenotfound = 0;
        size_t _framenum = 0;
        QStringList _filters;
        _filters << "*.mp4" << "*.webm" << "*.avi";
        const QString input_directory = _cmdparser.get<cv::String>("videodir").c_str();
        const std::vector<QString> absolute_paths = find_all_subdirs_in_path(_cmdparser.get<cv::String>("videodir").c_str());
        for(const auto & absolute_subdir_name : absolute_paths) {
            QDir _indir(absolute_subdir_name);
            QString subdirname = absolute_subdir_name.section(input_directory,1);
            QStringList _fileslist = _indir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);
            qInfo("There is %u videos has been found in the '%s'", static_cast<uint>(_fileslist.size()), subdirname.toUtf8().constData());
            QString target_output_path = _outdir.absolutePath();
            if(_fileslist.size() > 0 && _preservesubdirnames) {
                target_output_path = target_output_path.append("/%1").arg(subdirname);
                _outdir.mkpath(target_output_path);
            }
            for(int i = 0; i < _fileslist.size(); ++i) {
                cv::VideoCapture videocapture;
                const cv::String filename = _indir.absoluteFilePath(_fileslist.at(i)).toStdString();
                if(videocapture.open(filename)) {
                    qInfo(" - video file '%s' has been opened successfully", filename.c_str());
                    cv::Mat frame;
                    unsigned long framenum = 0;
                    while(videocapture.read(frame)) {
                        if(framenum % strobe == 0) {
                            std::vector<cv::Rect> bboxes = facedetector->detectFaces(frame);
                            std::vector<std::vector<cv::Point2f>> _faces;
                            facelandmarker->fit(frame, bboxes, _faces);
                            if(_faces.size() != 0) {
                                qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
                                const QString filename_woext = _fileslist.at(i).section('.',0,0);
                                for(size_t j = 0; j < _faces.size(); ++j) {
                                    const cv::Mat _facepatch = cv::ofrt::Facemark::extractFace(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                                    if(_visualize) {
                                        cv::imshow("Probe",_facepatch);
                                        cv::waitKey(1);
                                    }
                                    if(!_preservefilenames)
                                        cv::imwrite(QString("%1/%2.%3").arg(target_output_path,QUuid::createUuid().toString(),extension).toUtf8().constData(),_facepatch);
                                    else if( _faces.size() == 1)
                                        cv::imwrite(QString("%1/%2.%3").arg(target_output_path,filename_woext,extension).toUtf8().constData(),_facepatch);
                                    else
                                        cv::imwrite(QString("%1/%2_%3.%4").arg(target_output_path,QString::number(j),filename_woext,extension).toUtf8().constData(),_facepatch);
                                }
                            } else
                                qInfo("frame %lu - no faces", framenum);
                        }
                        framenum++;
                    }
                } else {
                    qInfo(" - !!! can not open '%s' !!!", filename.c_str());
                }
            }
        }
        qInfo("Done, frames without faces: %lu / %lu", static_cast<unsigned long>(_facenotfound), static_cast<unsigned long>(_framenum));
    } else if(_cmdparser.has("videofile")) {
        cv::VideoCapture videocapture;
        const cv::String filename = _cmdparser.get<cv::String>("videofile");
        if(videocapture.open(filename)) {
            qInfo("Video file has been opened successfully");
            cv::Mat frame;
            unsigned long framenum = 0;
            while(videocapture.read(frame)) {
                if(framenum % strobe == 0) {
                    std::vector<cv::Rect> bboxes = facedetector->detectFaces(frame);
                    std::vector<std::vector<cv::Point2f>> _faces;
                    facelandmarker->fit(frame, bboxes, _faces);
                    if(_faces.size() != 0) {
                        qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
                        for(size_t j = 0; j < _faces.size(); ++j) {
                            const cv::Mat _facepatch = cv::ofrt::Facemark::extractFace(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                            if(_visualize) {
                                cv::imshow("Probe",_facepatch);
                                cv::waitKey(1);
                            }
                            cv::imwrite(QString("%1/%2.%3").arg(_outdir.absolutePath(),QUuid::createUuid().toString(),extension).toStdString(),_facepatch);
                        }
                    } else
                        qInfo("frame %lu - no faces", framenum);
                }
                framenum++;
            }
        } else {
            qInfo("Can not open '%s'! Abort...", filename.c_str());
            return 5;
        }
    } else if(_cmdparser.has("inputdir")) {
        size_t _facenotfound = 0;
        size_t _totalfiles = 0;
        QStringList _filters;
        _filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
        const QString input_directory = _cmdparser.get<cv::String>("inputdir").c_str();
        const std::vector<QString> absolute_paths = find_all_subdirs_in_path(_cmdparser.get<cv::String>("inputdir").c_str());
        for(const auto & absolute_subdir_name : absolute_paths) {
            QDir _indir(absolute_subdir_name);
            QString subdirname = absolute_subdir_name.section(input_directory,1);
            QStringList _fileslist = _indir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);
            qInfo("There is %u pictures has been found in the '%s'", static_cast<uint>(_fileslist.size()), subdirname.toUtf8().constData());
            QString target_output_path = _outdir.absolutePath();
            if(_fileslist.size() > 0 && _preservesubdirnames) {
                target_output_path = target_output_path.append("/%1").arg(subdirname);
                _outdir.mkpath(target_output_path);
            }
            for(int i = 0; i < _fileslist.size(); ++i) {
                /*if(QFileInfo(_indir.absoluteFilePath(_fileslist.at(i))).size()/1024 > 7) {
                    qInfo("%d) %s - skipped", i, _fileslist.at(i).toUtf8().constData());
                    continue;
                }*/
                _totalfiles++;
                cv::Mat _tmpmat = cv::imread(_indir.absoluteFilePath(_fileslist.at(i)).toLocal8Bit().constData());

                /*float resize = cvrng.uniform(0.15f,0.6f);
                cv::resize(_tmpmat,_tmpmat,cv::Size(),resize,resize);
                _tmpmat = jitterimage(_tmpmat,cvrng,cv::Size(0,0),0.05,0.05,15,cv::BORDER_REFLECT,cv::Scalar(0),false);
                _tmpmat *= cvrng.uniform(0.1f,0.9f);

                if(cvrng.uniform(0.0f,1.0f) < 0.5f) {
                    _tmpmat = applyMotionBlur(_tmpmat,90.0f*cvrng.uniform(0.0f,1.0f),cvrng.uniform(2,3));
                    _tmpmat = applyMotionBlur(_tmpmat,90.0f*cvrng.uniform(0.0f,1.0f),cvrng.uniform(2,3));
                }
                if(cvrng.uniform(0.0f,1.0f) < 1.5f)
                    _tmpmat = addNoise(_tmpmat,cvrng,0,cvrng.uniform(16,80));

                std::vector<unsigned char> _bytes;
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(cvrng.uniform(65,99));
                cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
                _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);*/


                /*if(cvrng.uniform(0.0f,1.0f) < 0.5f)
                    _tmpmat = posterize(_tmpmat, cvrng.uniform(1,32));*/



                /*
                // 0.175 - 45 - 85 %, 0.35 - 90 - 25 %, 0.70 - 180 - 15 %
                int max_jpeg_quality = (int)(100.0f*(0.08f / resize));

                std::vector<unsigned char> _bytes;
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
                compression_params.push_back(cvrng.uniform(5,max_jpeg_quality));
                cv::imencode("*.jpg",_tmpmat,_bytes,compression_params);
                _tmpmat = cv::imdecode(_bytes,cv::IMREAD_UNCHANGED);
                */


                cv::Mat _imgmat = _tmpmat;
                if(!_imgmat.empty()) {
                    std::vector<cv::Rect> bboxes = facedetector->detectFaces(_imgmat);
                    std::vector<std::vector<cv::Point2f>> _faces;
                    facelandmarker->fit(_imgmat, bboxes, _faces);
                    /*cv::Ptr<cv::ofrt::YuNetFaceDetector> yundet = facedetector.dynamicCast<cv::ofrt::YuNetFaceDetector>();
                    const std::vector<std::vector<cv::Point2f>> _faces = yundet->detectLandmarks(_imgmat);*/
                    if(_faces.size() == 0) {
                        qInfo("%d) %s - could not find any faces!", i, _fileslist.at(i).toUtf8().constData());
                        _facenotfound++;
                    } else {
                        qInfo("%d) %s - %d face/s found", i, _fileslist.at(i).toUtf8().constData(), static_cast<int>(_faces.size()));
                        const QString filename_woext = _fileslist.at(i).section('.',0,0);
                        for(size_t j = 0; j < qMax(_faces.size(),size_t(1)); ++j) {
                            //const cv::Mat _facepatch = cv::ofrt::Facemark::extractFace(_imgmat,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                            const cv::Mat _facepatch = cv::ofrt::FaceClassifier::extractFacePatch(_imgmat,_faces[j],_targetsize,cv::INTER_LINEAR);
                            if(_visualize) {
                                cv::imshow("Probe",_facepatch);
                                cv::waitKey(1);
                            }
                            if(!_preservefilenames)
                                cv::imwrite(QString("%1/%2.%3").arg(target_output_path,QUuid::createUuid().toString(),extension).toUtf8().constData(),_facepatch);
                            else if( _faces.size() == 1)
                                cv::imwrite(QString("%1/%2.%3").arg(target_output_path,filename_woext,extension).toUtf8().constData(),_facepatch);
                            else
                                cv::imwrite(QString("%1/%2_%3.%4").arg(target_output_path,QString::number(j),filename_woext,extension).toUtf8().constData(),_facepatch);
                        }
                    }
                } else {
                    qInfo("%d) %s - could not decode image!", i, _fileslist.at(i).toUtf8().constData());
                }
            }
        }
        qInfo("Done, images without faces: %lu / %lu", static_cast<unsigned long>(_facenotfound), static_cast<unsigned long>(_totalfiles));
    } else {
        qWarning("You have not specified any input. Nor videofile nor directory! Abort...");
        return 1;
    }
    return 0;
}


std::vector<QString> find_all_subdirs_in_path(const QString &path) {
    std::vector<QString> all_levels_subdirs_list;
    QDir dir(path);
    const QStringList subdirsnames = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    if(subdirsnames.size() > 0) {
        for(const QString &subdirname: subdirsnames) {
            std::vector<QString> subdirs_list = find_all_subdirs_in_path(QString("%1/%2").arg(path,subdirname));
            all_levels_subdirs_list.insert(all_levels_subdirs_list.end(),subdirs_list.begin(),subdirs_list.end());
        }
    }
    all_levels_subdirs_list.push_back(path);
    return all_levels_subdirs_list;
}

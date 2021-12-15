#include <QStringList>
#include <QUuid>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "cnnfacedetector.h"
#include "facemarkcnn.h"

#include "facextractionutils.h"

const cv::String _options = "{help h               |                        | this help                                                     }"
                            "{inputdir i           |                        | input directory with images                                   }"
                            "{videofile            |                        | input videofile, if used will be processed instead of inputdir}"
                            "{outputdir o          |                        | output directory with images                                  }"
                            "{facedetmodel m       | res10_300x300_ssd_iter_140000_fp16.caffemodel | face detector model                    }"
                            "{facedetdscr d        | deploy_lowres.prototxt | face detector description                                     }"
                            "{confthresh           | 0.3                    | confidence threshold for the face detector                    }"
                            "{facelandmarksmodel l | facelandmarks_net.dat  | face landmarks model (68 points)                              }"
                            "{targeteyesdistance   | 90.0                   | target distance between eyes                                  }"
                            "{targetwidth          | 300                    | target image width                                            }"
                            "{targetheight         | 400                    | target image height                                           }"
                            "{h2wshift             | 0                      | additional horizontal shift to face crop in portion of target width}"
                            "{v2hshift             | 0                      | additional vertical shift to face crop in portion of target height }"
                            "{rotate               | true                   | apply rotation to make eyes-line horizontal aligned           }"
                            "{videostrobe          | 30                     | only each videostrobe frame will be processed                 }"
                            "{visualize v          | false                  | enable/disable visualization option                           }"
                            "{preservefilenames    | false                  | enable/disable filenames preservation                         }"
                            "{preservesubdirnames  | false                  | enable/disable subdirs preservation                           }";

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
    cv::Ptr<cv::ofrt::FaceDetector> facedetector = cv::ofrt::CNNFaceDetector::createDetector(_cmdparser.get<std::string>("facedetdscr"),
                                                                                             _cmdparser.get<std::string>("facedetmodel"),
                                                                                             _cmdparser.get<float>("confthresh"));
    cv::Ptr<cv::face::Facemark> facelandmarker = cv::face::createFacemarkCNN();
    facelandmarker->loadModel(_cmdparser.get<std::string>("facelandmarksmodel"));


    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    float _targeteyesdistance = _cmdparser.get<float>("targeteyesdistance");
    const bool _visualize = _cmdparser.get<bool>("visualize");
    const bool _preservefilenames = _cmdparser.get<bool>("preservefilenames");
    const bool _preservesubdirnames = _cmdparser.get<bool>("preservesubdirnames");
    const float h2wshift = _cmdparser.get<float>("h2wshift");
    const float v2hshift = _cmdparser.get<float>("v2hshift");
    const bool rotate = _cmdparser.get<bool>("rotate");

    if(_cmdparser.has("videofile")) {
        cv::VideoCapture videocapture;
        const cv::String filename = _cmdparser.get<cv::String>("videofile");
        if(videocapture.open(filename)) {
            qInfo("Video file has been opened successfully");
            cv::Mat frame;
            unsigned long framenum = 0;
            const unsigned long strobe = static_cast<unsigned long>(_cmdparser.get<unsigned int>("videostrobe"));
            while(videocapture.read(frame)) {
                if(framenum % strobe == 0) {
                    const std::vector<std::vector<cv::Point2f>> _faces = detectFacesLandmarks(frame,facedetector,facelandmarker);
                    if(_faces.size() != 0) {
                        qInfo("frame # %lu - %d face/s found", framenum, static_cast<int>(_faces.size()));
                        for(size_t j = 0; j < _faces.size(); ++j) {
                            const cv::Mat _facepatch = extractFacePatch(frame,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                            if(_visualize) {
                                cv::imshow("Probe",_facepatch);
                                cv::waitKey(1);
                            }
                            cv::imwrite(QString("%1/%2.jpg").arg(_outdir.absolutePath(),QUuid::createUuid().toString()).toUtf8().constData(),_facepatch);
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
        const std::vector<QString> absolute_paths = find_all_subdirs_in_path(_cmdparser.get<cv::String>("inputdir").c_str());
        for(const auto & absolute_subdir_name : absolute_paths) {
            QDir _indir(absolute_subdir_name);
            QString subdirname = absolute_subdir_name.section('/',-1);
            QStringList _fileslist = _indir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);
            qInfo("There is %d pictures has been found in the '%s'", _fileslist.size(), subdirname.toUtf8().constData());
            QString target_output_path = _outdir.absolutePath();
            if(_fileslist.size() > 0 && _preservesubdirnames) {
                _outdir.mkdir(subdirname);
                target_output_path = target_output_path.append("/%1").arg(subdirname);
            }
            for(int i = 0; i < _fileslist.size(); ++i) {
                _totalfiles++;
                cv::Mat _imgmat = cv::imread(_indir.absoluteFilePath(_fileslist.at(i)).toLocal8Bit().constData());
                if(!_imgmat.empty()) {
                    const std::vector<std::vector<cv::Point2f>> _faces = detectFacesLandmarks(_imgmat,facedetector,facelandmarker);
                    if(_faces.size() == 0) {
                        qInfo("%d) %s - could not find any faces!", i, _fileslist.at(i).toUtf8().constData());
                        _facenotfound++;
                    } else {
                        qInfo("%d) %s - %d face/s found", i, _fileslist.at(i).toUtf8().constData(), static_cast<int>(_faces.size()));
                        for(size_t j = 0; j < _faces.size(); ++j) {
                            const cv::Mat _facepatch = extractFacePatch(_imgmat,_faces[j],_targeteyesdistance,_targetsize,h2wshift,v2hshift,rotate);
                            if(_visualize) {
                                cv::imshow("Probe",_facepatch);
                                cv::waitKey(1);
                            }
                            if(!_preservefilenames)
                                cv::imwrite(QString("%1/%2.jpg").arg(target_output_path,QUuid::createUuid().toString()).toUtf8().constData(),_facepatch);
                            else if( _faces.size() == 1)
                                cv::imwrite(QString("%1/%2").arg(target_output_path,_fileslist.at(i)).toUtf8().constData(),_facepatch);
                            else
                                cv::imwrite(QString("%1/%2_%3").arg(target_output_path,QString::number(j),_fileslist.at(i)).toUtf8().constData(),_facepatch);
                        }
                    }
                } else {
                    qInfo("%d) %s - could not decode image!", i, _fileslist.at(i).toUtf8().constData());
                }
            }
        }
        qInfo("Done, percentage of images without faces: %lu / %lu", static_cast<unsigned long>(_facenotfound), static_cast<unsigned long>(_totalfiles));
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
    } else
        all_levels_subdirs_list.push_back(path);
    return all_levels_subdirs_list;
}

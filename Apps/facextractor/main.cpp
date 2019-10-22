#include <QStringList>
#include <QUuid>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "cnnfacedetector.h"
#include "multyfacetracker.h"

const cv::String _options = "{help h            |     | this help                           }"
                            "{inputdir i        |     | input directory with images         }"
                            "{outputdir o       |     | output directory with images        }"
                            "{facedetmodel m    |     | face detector model                 }"
                            "{facedetdscr d     |     | face detector description           }"
                            "{confthresh        | 0.5 | confidence threshold for the face detector}"
                            "{targetwidth       | 150 | target image width                  }"
                            "{targetheight      | 150 | target image height                 }"
                            "{hportion          | 1.2 | target horizontal face portion      }"
                            "{vportion          | 1.2 | target vertical face portion        }"
                            "{visualize v       |false| enable/disable visualization option }";

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE, "Rus");
#endif

    cv::CommandLineParser _cmdparser(argc,argv,_options);
    if(_cmdparser.has("help")) {
        _cmdparser.printMessage();
        return 0;
    }

    if(!_cmdparser.has("inputdir")) {
        qWarning("You have not specified an input directory! Abort...");
        return 1;
    }
    QDir _indir(_cmdparser.get<cv::String>("inputdir").c_str());
    QStringList _filters;
    _filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    QStringList _fileslist = _indir.entryList(_filters, QDir::Files | QDir::NoDotAndDotDot);
    qInfo("There is %d pictures has been found in the input directory", _fileslist.size());

    if(!_cmdparser.has("outputdir")) {
        qWarning("You have not specified output directory! Abort...");
        return 2;
    }
    QDir _outdir(_cmdparser.get<cv::String>("outputdir").c_str());
    if(_outdir.exists() == false) {
        _outdir.mkpath(_cmdparser.get<cv::String>("outputdir").c_str());
        _outdir.cd(_cmdparser.get<cv::String>("outputdir").c_str());
    }

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

    const cv::Size _targetsize(_cmdparser.get<int>("targetwidth"),_cmdparser.get<int>("targetheight"));
    const float _hp = _cmdparser.get<float>("hportion"), _vp = _cmdparser.get<float>("vportion");
    const bool _visualize = _cmdparser.get<bool>("visualize");
    size_t _facenotfound = 0;
    for(int i = 0; i < _fileslist.size(); ++i) {
        cv::Mat _imgmat = cv::imread(_indir.absoluteFilePath(_fileslist.at(i)).toLocal8Bit().constData());
        const std::vector<cv::Rect> _facesboxes = facedetector->detectFaces(_imgmat);
        if(_facesboxes.size() == 0) {
            qInfo("%d) %s - could not find any faces!", i, _fileslist.at(i).toUtf8().constData());
            _facenotfound++;
        } else {
            qInfo("%d) %s - %d face/s found", i, _fileslist.at(i).toUtf8().constData(), static_cast<uint>(_facesboxes.size()));
            const cv::Rect _framerect = cv::Rect(0,0,_imgmat.cols,_imgmat.rows);
            for(size_t j = 0; j < _facesboxes.size(); ++j) {
                const cv::Rect _rect = cv::Rect(_facesboxes[j].x - _facesboxes[j].width*(_hp - 1.0f)/2.0f,
                                                _facesboxes[j].y - _facesboxes[j].height*(_vp - 1.0f)/2.0f,
                                                _facesboxes[j].width*_hp, _facesboxes[j].height*_vp) & _framerect;
                cv::Mat _facemat = cv::ofrt::MultyFaceTracker::__cropInsideFromCenterAndResize(_imgmat(_rect),_targetsize);
                if(_visualize) {
                    cv::imshow("Probe",_facemat);
                    cv::waitKey(1);
                }
                cv::imwrite(QString("%1/%2.jpg").arg(_outdir.absolutePath(),QUuid::createUuid().toString()).toUtf8().constData(),_facemat);
            }
        }
    }
    qInfo("Done, percentage of images without faces: %d / %d", _fileslist.size() - _facenotfound, _fileslist.size());

    return 0;
}

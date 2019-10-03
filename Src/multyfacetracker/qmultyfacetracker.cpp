#include "qmultyfacetracker.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

QMultyFaceTracker::QMultyFaceTracker(const cv::Ptr<cv::ofrt::FaceDetector> &_cvptrfacedet, uint _maxfaces, QObject *parent) : QObject(parent),
    targetSize(cv::Size(150,150)),
    visualization(false)
{
    qRegisterMetaType<cv::RotatedRect>("cv::RotatedRect");
    qRegisterMetaType<cv::String>("cv::String");

    multyfacetracker.setFaceDetector(_cvptrfacedet,static_cast<size_t>(_maxfaces));
}

void QMultyFaceTracker::enrollImage(const cv::Mat &inputImg)
{
    if(!inputImg.empty()) {
        std::vector<std::pair<size_t,cv::Mat>> vfaces = multyfacetracker.getResizedFaceImages(inputImg,targetSize,1);
        for(std::size_t i = 0; i < vfaces.size(); i++) {
            cv::ofrt::TrackedFace *_ptrackedface = multyfacetracker.at(vfaces[i].first);
            if((_ptrackedface->getMetaId() < 0) &&
               (_ptrackedface->getUnknownInRow() <= 7) && // this should decrease load of identificatio nserver
               (_ptrackedface->getPosted2Srv() == false)) {
                emit faceWithoutLabelFound(vfaces[i].second, _ptrackedface->getUuid());
                _ptrackedface->setPosted2Srv(true);
            }
        }
    }

    // visualization
    if(visualization) {
        static int64 _ticks;
        static int64 _temp;
        _temp = cv::getTickCount();
        if(inputImg.empty() == false) {

            cv::Mat _imgmat = inputImg;
            float _pointsize = _imgmat.cols * 0.000390625f + 0.25;
            int _fonttype = cv::FONT_HERSHEY_SIMPLEX;
            int _thickness = std::floor( _imgmat.cols/1280.0f );
            if(_thickness < 1)
                _thickness = 1;
            for(size_t i = 0 ; i < multyfacetracker.maxFaces(); i++) {
                const cv::ofrt::TrackedFace *_ptrackedface = multyfacetracker.at(i);
                if( _ptrackedface->getFramesTracked() > 0 ) {
                    const cv::Rect _rect = _ptrackedface->getRect(1);
                    cv::rectangle(_imgmat,_rect,cv::Scalar(0,255,0), _thickness, cv::LINE_AA);

                    const cv::Point _tl(_rect.x,_rect.y);
                    const cv::Point _bl(_rect.x,_rect.y+_rect.height);

                    // Draw face tracking info
                    std::string _ftinfo = std::string("FD ") + std::to_string(i) + std::string(" (uuid: ") + std::to_string(_ptrackedface->getUuid()) + std::string(")");
                    cv::putText(_imgmat, _ftinfo, _tl + cv::Point(4+_pointsize,20+_pointsize), _fonttype, _pointsize*0.75, cv::Scalar(0,0,0), _thickness, cv::LINE_AA);
                    cv::putText(_imgmat, _ftinfo, _tl + cv::Point(3+_pointsize,19+_pointsize), _fonttype, _pointsize*0.75, cv::Scalar(255,255,255), _thickness, cv::LINE_AA);
                    // Draw label info
                    cv::String _infostr = utf8cyr2utf8latin( _ptrackedface->getMetaInfo() );
                    cv::putText(_imgmat, _infostr, _tl - cv::Point(4+_pointsize,4+_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, cv::LINE_AA);
                    cv::putText(_imgmat, _infostr, _tl - cv::Point(4+3*_pointsize,4+3*_pointsize), _fonttype, _pointsize, _ptrackedface->getPosted2Srv() ? cv::Scalar(100,100,100) : cv::Scalar(0,255,0), _thickness, cv::LINE_AA);
                    // Draw distance
                    cv::String _confstr = "Dist.:" + cv::String(QString::number(_ptrackedface->getMetaId() > -1 ? _ptrackedface->getMetaDistance(): -1.0, 'f', 2).toLocal8Bit().constData());
                    cv::putText(_imgmat, _confstr, _bl - cv::Point(-4-_pointsize,4+_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, cv::LINE_AA);
                    cv::putText(_imgmat, _confstr, _bl - cv::Point(-4-3*_pointsize,4+3*_pointsize), _fonttype, _pointsize, _ptrackedface->getPosted2Srv() ? cv::Scalar(100,100,100) : cv::Scalar(255,255,255), _thickness, cv::LINE_AA);
                    // Draw class label
                    cv::String _labelstr = "ID:" + cv::String(QString::number(_ptrackedface->getMetaId()).toLocal8Bit().constData());
                    cv::putText(_imgmat, _labelstr, _bl + cv::Point(4+5*_pointsize,4+30*_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, cv::LINE_AA);
                    cv::putText(_imgmat, _labelstr, _bl + cv::Point(4+5*_pointsize,4+27*_pointsize), _fonttype, _pointsize, _ptrackedface->getPosted2Srv() ? cv::Scalar(100,100,100) : cv::Scalar(0,255,0), _thickness, cv::LINE_AA);
                }
            }
            // Control time
            double _time =  static_cast<double>(_temp - _ticks) * 1000.0 / cv::getTickFrequency();
            _ticks = _temp;
            cv::String _timestr = QString("%1x%2 %3 ms").arg(QString::number(_imgmat.cols),QString::number(_imgmat.rows),QString::number(_time,'f',1)).toStdString();
            cv::putText(_imgmat, _timestr, cv::Point(6,_imgmat.rows - 4), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, cv::LINE_AA);
            cv::putText(_imgmat, _timestr, cv::Point(5,_imgmat.rows - 5), _fonttype, _pointsize, cv::Scalar(255,255,255), _thickness, cv::LINE_AA);

            cv::namedWindow("Video probe", cv::WINDOW_NORMAL);
            cv::imshow("Video probe",_imgmat);
        }
    }
    emit frameProcessed();
}

void QMultyFaceTracker::setTargetFaceSize(const cv::Size &size)
{
    targetSize = size;
}

void QMultyFaceTracker::enableVisualization(bool _value)
{
    visualization = _value;
}


void QMultyFaceTracker::setLabelForTheFace(int _id, double _distance, const cv::String &_info, unsigned long _uuid)
{
    for(size_t i = 0; i < multyfacetracker.maxFaces(); ++i) {
        cv::ofrt::TrackedFace *_ptrackedface = multyfacetracker.at(i);
        if(_ptrackedface->getUuid() == _uuid) {
            _ptrackedface->setMetaData(_id,_distance,_info);
            _ptrackedface->setPosted2Srv(false);
            break;
        }
    }
}

cv::String utf8cyr2utf8latin(const cv::String &_cvcyrstr)
{
    QString str = QString::fromUtf8(_cvcyrstr.c_str());

    static const QString validChars = QString::fromUtf8("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890-_,.()[]{}<>~!@#$%^&*+=? ");
    static const QString rusUpper = QString::fromUtf8("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ");
    static const QString rusLower = QString::fromUtf8("абвгдеёжзийклмнопрстуфхцчшщыэюя");

    static const QStringList latUpper = QStringList() <<"A"<<"B"<<"V"<<"G"<<"D"<<"E"<<"Jo"<<"Zh"<<"Z"<<"I"<<"J"<<"K"<<"L"<<"M"<<"N"
                                                      <<"O"<<"P"<<"R"<<"S"<<"T"<<"U"<<"F"<<"H"<<"TS"<<"Ch"<<"Sh"<<"Sh"<<"I"<<"E"<<"Ju"<<"Ja";

    static const QStringList latLower = QStringList() <<"a"<<"b"<<"v"<<"g"<<"d"<<"e"<<"jo"<<"zh"<<"z"<<"i"<<"j"<<"k"<<"l"<<"m"<<"n"
                                                      <<"o"<<"p"<<"r"<<"s"<<"t"<<"u"<<"f"<<"h"<<"ts"<<"ch"<<"sh"<<"sh"<<"i"<<"e"<<"ju"<<"ja";

    QString _outlatinstr;
    int i, rU, rL;
    for(i = 0; i < str.size(); i++) {
        if( validChars.contains(str[i]) ){
            _outlatinstr.append(str[i]);
        } else {
            rL = rusLower.indexOf(str[i],0,Qt::CaseSensitive);
            if(rL > -1) {
                _outlatinstr.append(latLower[rL]);
            } else {
                rU = rusUpper.indexOf(str[i],0,Qt::CaseSensitive);
                if(rU > -1) {
                    _outlatinstr.append(latUpper[rU]);
                }
            }
        }
    }
    return cv::String(_outlatinstr.toLocal8Bit().constData());
}

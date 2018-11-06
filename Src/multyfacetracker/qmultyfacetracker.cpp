#include "qmultyfacetracker.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <QDateTime>

QMultyFaceTracker::QMultyFaceTracker(uint _maxfaces, QObject *parent) : QObject(parent)
{
    qRegisterMetaType<cv::RotatedRect>("cv::RotatedRect");
    qRegisterMetaType<cv::String>("cv::String");

    setMaxFaces(_maxfaces);
}

void QMultyFaceTracker::enrollImage(const cv::Mat &inputImg)
{
    if(!inputImg.empty()) {
        std::vector<cv::Mat> v_faces = m_tracker.getResizedFaceImages(inputImg, m_targetSize);        
        for(std::size_t i = 0; i < v_faces.size(); i++) {            
            //emit faceFound(v_faces[i]);
            FaceTracker *_pfacetracker = m_tracker[static_cast<int>(i)];
            if((_pfacetracker->getMetaID() == -1) &&
               (_pfacetracker->getFaceTrackedFrames() > _pfacetracker->getHistoryLength()) &&
               (_pfacetracker->inProcessing() == false)) {
                _pfacetracker->setInProcessing(true);
                emit faceWithoutLabelFound(v_faces[i], _pfacetracker->getQuuid());
            }
        }
    }

    // visualization
    if(f_visualization) {
        static int64 _ticks;
        static int64 _temp;
        _temp = cv::getTickCount();
        if(inputImg.empty() == false) {

            cv::Mat _imgmat = inputImg;
            float _pointsize = _imgmat.cols * 0.000390625f + 0.25;
            int _fonttype = CV_FONT_HERSHEY_SIMPLEX;
            int _thickness = (int)std::floor( _imgmat.cols/1280.0f );
            for(size_t i = 0 ; i < m_tracker.getMaxFaces(); i++) {
                const FaceTracker *_ptracker = m_tracker.at(static_cast<int>(i));
                if( _ptracker->getFaceRotatedRect().size.area() > 1.0f /*_ptracker->getMetaID() > -1*/) {
                    cv::RotatedRect _rrect = _ptracker->getFaceRotatedRect();
                    cv::Point2f _vertices[4];
                    _rrect.points(_vertices);

                    for(int j = 0; j < 4; j++)
                        cv::line(_imgmat, _vertices[j], _vertices[(j+1)%4], cv::Scalar(0,255,0), _thickness, CV_AA);

                    // Draw label info
                    cv::String _infostr = utf8cyr2utf8latin( _ptracker->getMetaInfo() );
                    cv::putText(_imgmat, _infostr, _vertices[1] - cv::Point2f(4.f+_pointsize,4.f+_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, CV_AA);
                    cv::putText(_imgmat, _infostr, _vertices[1] - cv::Point2f(4.f+3.f*_pointsize,4.f+3.f*_pointsize), _fonttype, _pointsize, cv::Scalar(0,255,0), _thickness, CV_AA);
                    // Draw confidence
                    cv::String _confstr = "Dist.:" + cv::String(QString::number(_ptracker->getMetaID() > -1 ? _ptracker->getMetaConfidence(): -1.0, 'f', 2).toLocal8Bit().constData());
                    cv::putText(_imgmat, _confstr, _vertices[0] - cv::Point2f(-4.f-_pointsize,4.f+_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, CV_AA);
                    cv::putText(_imgmat, _confstr, _vertices[0] - cv::Point2f(-4.f-3.f*_pointsize,4.f+3.f*_pointsize), _fonttype, _pointsize, cv::Scalar(255,255,255), _thickness, CV_AA);
                    // Draw class label
                    cv::String _labelstr = "ID:" + cv::String(QString::number(_ptracker->getMetaID()).toLocal8Bit().constData());
                    cv::putText(_imgmat, _labelstr, _vertices[1] + cv::Point2f(4.f+5.f*_pointsize,4.f+30.f*_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, CV_AA);
                    cv::putText(_imgmat, _labelstr, _vertices[1] + cv::Point2f(4.f+5.f*_pointsize,4.f+27.f*_pointsize), _fonttype, _pointsize, cv::Scalar(0,0,255), _thickness, CV_AA);
                }
            }
            // Control time
            double _time =  static_cast<double>(_temp - _ticks) * 1000.0 / cv::getTickFrequency();
            _ticks = _temp;
            cv::String _timestr = QString("%1x%2 %3 ms").arg(QString::number(_imgmat.cols),QString::number(_imgmat.rows),QString::number(_time,'f',1)).toStdString();
            cv::putText(_imgmat, _timestr, cv::Point(6,_imgmat.rows - 4), _fonttype, _pointsize, cv::Scalar(0,0,0), _thickness, CV_AA);
            cv::putText(_imgmat, _timestr, cv::Point(5,_imgmat.rows - 5), _fonttype, _pointsize, cv::Scalar(255,255,255), _thickness, CV_AA);

            cv::namedWindow("Video probe", CV_WINDOW_NORMAL);
            cv::imshow("Video probe",_imgmat);
        }
    }
    emit frameProcessed();
}

bool QMultyFaceTracker::setFaceClassifier(cv::CascadeClassifier *pointer)
{
    return m_tracker.setFaceClassifier(pointer);
}

void QMultyFaceTracker::setDlibFaceShapePredictor(dlib::shape_predictor *pointer)
{
    m_tracker.setDlibFaceShapePredictor(pointer);
}

bool QMultyFaceTracker::setEyeClassifier(cv::CascadeClassifier *pointer)
{
    return m_tracker.setEyeClassifier(pointer);
}

void QMultyFaceTracker::setTargetFaceSize(const cv::Size &size)
{
    m_targetSize = size;
}

void QMultyFaceTracker::setMaxFaces(uint value)
{
    m_tracker.setMaxFaces(value);
}

void QMultyFaceTracker::setFaceRectPortions(float _xP, float _yP)
{
    m_tracker.setFaceRectPortions(_xP,_yP);
}

void QMultyFaceTracker::setFaceRectShifts(float _xShift, float _yShift)
{
    m_tracker.setFaceRectShifts(_xShift, _yShift);
}

void QMultyFaceTracker::setFaceAlignmentMethod(FaceTracker::AlignMethod _method)
{
    m_tracker.setFaceAlignMethod(_method);
}

void QMultyFaceTracker::setLabelForTheFace(int _id, double _confidence, const cv::String &_info, const QUuid &_quuid)
{
    FaceTracker *_ptracker;
    for(size_t i = 0; i < m_tracker.getMaxFaces(); ++i) {
        _ptracker = m_tracker[static_cast<int>(i)];
        if(_ptracker->getQuuid() == _quuid) {
            _ptracker->setMetaData(_id,_confidence,_info);
            _ptracker->setInProcessing(false);
            break;
        }
    }
}

size_t QMultyFaceTracker::getMaxFaces() const
{
    return m_tracker.getMaxFaces();
}

void QMultyFaceTracker::setVisualization(bool _value)
{
    f_visualization = _value;
}

void QMultyFaceTracker::setVerbose(bool _enable)
{
    f_verbose = _enable;
}

int QMultyFaceTracker::getRecognizerid() const
{
    return recognizerid;
}

void QMultyFaceTracker::setRecognizerid(int value)
{
    recognizerid = value;
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

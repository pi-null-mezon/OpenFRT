#include "qvideocapture.h"

QVideoCapture::QVideoCapture(QObject *parent) : QObject(parent),
    pt_timer(nullptr),
    m_timerintervalms(15),
    flipFlag(false)
{
    qRegisterMetaType<cv::Mat>("cv::Mat");    
}

QVideoCapture::~QVideoCapture()
{
    close();
    if(pt_timer != nullptr)
        delete pt_timer;
}

bool QVideoCapture::openDevice(int _id)
{
    m_sourcetype = SourceType::Device;
    m_sourcename = QString::number(_id).toUtf8().constData();
    return m_videocapture.open(_id);
}

bool QVideoCapture::openURL(const cv::String &_url)
{
    m_sourcetype = SourceType::Url;
    m_sourcename = _url;
    return m_videocapture.open(_url);
}

bool QVideoCapture::openFile(const cv::String &_filename)
{
    m_sourcetype = SourceType::File;
    m_sourcename = _filename;
    return m_videocapture.open(_filename);
}

void QVideoCapture::close()
{
    pause();
    m_videocapture.release();
}

void QVideoCapture::pause()
{
    if(pt_timer != nullptr)
        pt_timer->stop();
    else
        qWarning("QVideoCapture: not initialized!");
}

void QVideoCapture::resume()
{
    if(pt_timer != nullptr)
        pt_timer->start();
    else
        qWarning("QVideoCapture: not initialized!");
}

void QVideoCapture::setCaptureProps(int _width, int _height, int _fps)
{
    if(m_videocapture.isOpened()) {
        if(m_videocapture.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(_width)) == false)
            qWarning("QVideoCapture: can not set width to %d", _width);
        if(m_videocapture.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(_height)) == false)
            qWarning("QVideoCapture: can not set height to %d", _height);
        if(m_videocapture.set(cv::CAP_PROP_FPS, static_cast<double>(_fps)) == false)
            qWarning("QVideoCapture: can not set fps to %d", _fps);
        if(pt_timer != nullptr)
            pt_timer->setInterval( 1000.0 / _fps );
        else {
            m_timerintervalms = 1000.0 / _fps;
        }
    }
}

void QVideoCapture::init()
{
    if(pt_timer == nullptr) {
        pt_timer = new QTimer();
        pt_timer->setTimerType(Qt::PreciseTimer);
        pt_timer->setInterval(m_timerintervalms);
        connect(pt_timer, SIGNAL(timeout()), this, SLOT(__readFrame()));
    }
}

void QVideoCapture::__readFrame()
{
    if(m_videocapture.isOpened()) {
        cv::Mat _framemat;
        if(m_videocapture.read(_framemat)) {
            if(!_framemat.empty()) {
                if(flipFlag == false) {
                    emit frameUpdated(_framemat);
                } else {
                    cv::flip(_framemat, _framemat, -1); // flip both horizontally and vertically
                    emit frameUpdated(_framemat);
                }
            }
        } else {
            qInfo("\n!Warning! Video source disconnected! Can not read frames!\n");
            close();
            if(m_sourcetype == SourceType::Url) {
                qInfo("Application will try to reopen URL: %s", m_sourcename.c_str());
                close();
                unsigned long _attempts = 0;
                bool _reopened = false;
                do {
                    _reopened = openURL(m_sourcename);
                    qInfo("Attempt #%lu - %s", _attempts++, _reopened ? "success" : "fail");
                } while(_reopened == false);
                QTimer::singleShot(500, this, SLOT(resume()));
            }
        }
    }
}

bool QVideoCapture::getFlipFlag() const
{
    return flipFlag;
}

void QVideoCapture::setFlipFlag(bool value)
{
    flipFlag = value;
}

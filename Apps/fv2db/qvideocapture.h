#ifndef QVIDEOCAPTURE_H
#define QVIDEOCAPTURE_H

#include <QObject>
#include <QTimer>

#include <opencv2/opencv.hpp>

class QVideoCapture : public QObject
{    
    Q_OBJECT

public:
    enum SourceType {Device, File, Url};
    QVideoCapture(QObject *parent=nullptr);
    ~QVideoCapture();
    bool openDevice(int _id);
    bool openURL(const cv::String &_url);
    bool openFile(const cv::String &_filename);
    void setCaptureProps(int _width, int _height, int _fps);

    bool getFlipFlag() const;
    void setFlipFlag(bool value);

public slots:
    void init();
    void close();
    void pause();
    void resume();

signals:
    void frameUpdated(const cv::Mat &_matframe);

private slots:
    void __readFrame();

private:
    cv::VideoCapture m_videocapture;
    QTimer *pt_timer;
    int m_timerintervalms;
    bool flipFlag = false;
    SourceType m_sourcetype;
    cv::String m_sourcename;
};

#endif // QVIDEOCAPTURE_H

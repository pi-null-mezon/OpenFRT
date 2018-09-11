#ifndef QVIDEOLOCKER_H
#define QVIDEOLOCKER_H

#include <QObject>

#include <opencv2/core.hpp>

class QVideoLocker : public QObject
{
    Q_OBJECT
public:
    explicit QVideoLocker(QObject *parent = 0);

signals:
    void frameUpdated(const cv::Mat &_framemat, const cv::RotatedRect &_rrect);

public slots:
    void updateFrame(const cv::Mat &_framemat, const cv::RotatedRect &_rrect);
    void unlock();

private:
    void __lock();
    void __unlock();

    bool m_locked = false;
};

#endif // QVIDEOLOCKER_H

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
    void frameUpdated(const cv::Mat &_framemat, unsigned long _uuid);

public slots:
    void updateFrame(const cv::Mat &_framemat, unsigned long _uuid);
    void unlock();

private:
    void __lock();
    void __unlock();

    bool locked;
};

#endif // QVIDEOLOCKER_H

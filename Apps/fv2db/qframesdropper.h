#ifndef QFRAMESDROPPER_H
#define QFRAMESDROPPER_H

#include <QObject>
#include <opencv2/core.hpp>

class QFramesDropper : public QObject
{
    Q_OBJECT
public:
    explicit QFramesDropper(QObject *parent = nullptr);

signals:
    void frameUpdated(const cv::Mat &_frame);

public slots:
    void updateFrame(const cv::Mat &_frame);
    void passFrames();

private:
    bool drop;
};

#endif // QFRAMESDROPPER_H

#ifndef QIDENTIFICATIONTASKPOSTER_H
#define QIDENTIFICATIONTASKPOSTER_H

#include <QThread>

#include <opencv2/core.hpp>

class QIdentificationTaskPoster : public QThread
{
    Q_OBJECT
public:
    QIdentificationTaskPoster(const QString &_urlstr, const cv::Mat &_facemat, unsigned long _uuid, QObject *_parent=nullptr);

signals:
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, unsigned long _uuid);
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_img);

protected:
    void run();

private:
    QString         urlstr;
    cv::Mat         facemat;
    unsigned long   uuid;
};

#endif // QIDENTIFICATIONTASKPOSTER_H

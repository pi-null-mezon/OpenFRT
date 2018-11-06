#ifndef QIDENTIFICATIONTASKPOSTER_H
#define QIDENTIFICATIONTASKPOSTER_H

#include <QThread>
#include <QUuid>

#include <opencv2/core.hpp>

class QIdentificationTaskPoster : public QThread
{
    Q_OBJECT
public:
    QIdentificationTaskPoster(const QString &_urlstr, const cv::Mat &_facemat, const QUuid &_quuid, QObject *_parent=nullptr);

signals:
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, const QUuid &_quuid);
    void labelPredicted(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_img);

protected:
    void run();

private:
    QString urlstr;
    cv::Mat facemat;
    QUuid   quuid;
};

#endif // QIDENTIFICATIONTASKPOSTER_H

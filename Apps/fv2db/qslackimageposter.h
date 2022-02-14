#ifndef QSLACKIMAGEPOSTER_H
#define QSLACKIMAGEPOSTER_H

#include <QThread>

#include <opencv2/core.hpp>

void postImageIntoSlackChannel(const QString &_slackchannelid, const QString &_slacktoken, int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_cvmat);

class QSlackImagePoster : public QThread
{
    Q_OBJECT
public:
    QSlackImagePoster(const QString &_slackchannelid, const QString &_slacktoken, int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_cvmat);
    ~QSlackImagePoster();
protected:
    void run();

private:
    QString slackchannelid;
    QString slacktoken;
    int     label;
    double  distance;
    QString labelinfo;
    cv::Mat img;
};

#endif // QSLACKIMAGEPOSTER_H

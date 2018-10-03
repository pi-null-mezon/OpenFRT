#ifndef QSLACKCLIENT_H
#define QSLACKCLIENT_H

#include <QObject>
#include <QDateTime>

#include <opencv2/core.hpp>

class QSlackClient : public QObject
{
    Q_OBJECT

public:
    QSlackClient(QObject *_parent=0);

    void setSlackbottoken(const QString &value);
    QString getSlackbottoken() const;
    QString getSlackchannelid() const;
    void setSlackchannelid(const QString &value);

public slots:
    void enrollRecognition(int _label, double _distance, const cv::String &_labelInfo, const cv::Mat &_img);

private:
    QTime prevTime;
    int   prevLabel;
    unsigned int unrecfacesinrow;

    QString slackchannelid;
    QString slackbottoken;
};

#endif // QSLACKCLIENT_H

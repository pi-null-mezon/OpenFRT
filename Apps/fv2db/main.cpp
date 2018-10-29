#include <QCoreApplication>
#include <QTextStream>
#include <QSettings>
#include <QDateTime>
#include <QThread>

#include <QFile>
#include <QFileInfo>

#include "qmultyfacetracker.h"
#include "qvideocapture.h"
#include "qvideolocker.h"
#include "qslackclient.h"
#include "qslackimageposter.h"
#include "qfacerecognizer.h"
#include "qmongodbclient.h"

QFile *p_logfile = nullptr;

void logMessage(QtMsgType type, const QMessageLogContext &context, const QString &msg);

int main(int argc, char *argv[])
{
    #ifdef Q_OS_WIN
    setlocale(LC_CTYPE,"");
    #endif
    if(argc == 1) {
        qInfo("%s v.%s\n designed by %s in 2018", APP_NAME,APP_VERSION,APP_DESIGNER);
        qInfo(" -i[filename] - set settings filename");
        qInfo(" -l[filename] - set log file name");
        qInfo(" -d[int] - enumerator of the local videodevices to open");
        qInfo(" -s[url] - url of the videostream to process");
        qInfo(" -a[url] - url of the openirt web server (face identification API)");
        qInfo(" -e[url] - url of the MongoDB RESTfull interface to post events");
        qInfo(" -t[str] - auth token for MongoDB RESTfull interface");
        qInfo(" -p[str] - viewspot identificator");
        qInfo(" -v      - enable visualization");
        return 0;
    }

    QCoreApplication a(argc, argv);

    QString _logfilename, _videostreamurl, _identificationurl, _eveserverposturl, _mongodbaccesstoken, _viewspotid, _settingsfilename;
    int _videodeviceid = -1;
    bool _visualization = false;
    while((--argc > 0) && **(++argv) == '-')
        switch(*(++argv[0])) {
            case 'i':
                _settingsfilename = ++argv[0];
                break;
            case 'l':
                _logfilename = ++argv[0];
                break;
            case 'd':
                _videodeviceid = QString(++argv[0]).toInt();
                break;
            case 's':
                _videostreamurl = ++argv[0];
                break;
            case 'a':
                _identificationurl = ++argv[0];
                break;
            case 'v':
                _visualization = true;
                break;
            case 'e':
                _eveserverposturl = ++argv[0];
                break;
            case 't':
                _mongodbaccesstoken = ++argv[0];
                break;
            case 'p':
                _viewspotid = ++argv[0];
                break;
        }    
    // First let's check if user wants to use settings file
    if(!_settingsfilename.isEmpty()) {
        QFileInfo _sfinfo(_settingsfilename);
        if(_sfinfo.exists() == false) {
            qWarning("Can not find settings file %s! Abort...", _settingsfilename.toUtf8().constData());
            return 1;
        }
    }
    QSettings _settings(_settingsfilename,QSettings::IniFormat);
    // Check if logging needed
    if(!_logfilename.isEmpty()) {
        p_logfile = new QFile(_logfilename);
        if(p_logfile->open(QIODevice::Append)) {
            qInfo("All messages will be saved into logfile");
            qInstallMessageHandler(logMessage);
        } else {
            delete p_logfile;
            qWarning("Can not create log file, check your permissions! Abort...");
            return 2;
        }
    }
    // Let's try to open video source
    QVideoCapture _qvideocapture;
    _qvideocapture.setFlipFlag(_settings.value("Videoprops/Flip",false).toBool());
    if(_videodeviceid == -1)
        _videodeviceid = _settings.value("Videosource/Localdevice",-1).toInt();
    if(_videostreamurl.isEmpty())
        _videostreamurl = _settings.value("Videosource/Stream",QString()).toString();
    if(_videodeviceid > -1) {
        qInfo(" Trying to open video device %d",_videodeviceid);
        if(_qvideocapture.openDevice(_videodeviceid) == false) {
            qWarning("  Can not open videodevice! Abort...");
            return 3;
        } else {
            _qvideocapture.setCaptureProps(_settings.value("Videoprops/Width",640).toInt(),
                                           _settings.value("Videoprops/Height",360).toInt(),
                                           _settings.value("Videoprops/FPS",30).toInt());
            qInfo("  Success");
        }
    } else if(!_videostreamurl.isEmpty()) {
        qInfo(" Trying to open video stream %s", _videostreamurl.toUtf8().constData());
        if(_qvideocapture.openURL(_videostreamurl.toUtf8().constData()) == false) {
            qWarning("  Can not open video stream! Abort...");
            return 4;
        } else {
            qInfo("  Success");
        }
    } else {
        qWarning("  Video source has not been selected! Abort...");
        return 5;
    }
    // Let's check if user set viewspot identifier
    if(_viewspotid.isEmpty()) {
        _viewspotid = _settings.value("Location/ViewspotID","Unset").toString();
    }
    // Let's check if user want to save events into MongoDB
    if(_eveserverposturl.isEmpty()) {
        _eveserverposturl = _settings.value("MongoDB/URL",QString()).toString();
    }
    if(_mongodbaccesstoken.isEmpty()) {
        _mongodbaccesstoken = _settings.value("MongoDB/Token",QString()).toString();
    }

    // Ok, now the video source should be opened, let's prepare face tracker
    qInfo(" Trying to load face detection resources");
    QMultyFaceTracker _qmultyfacetracker(_settings.value("Facetracking/Maxfaces",7).toUInt());
    _qmultyfacetracker.setFaceRectPortions(_settings.value("Facetracking/FaceHPortion",1.35).toFloat(),
                                           _settings.value("Facetracking/FaceVPortion",1.75).toFloat());
    _qmultyfacetracker.setTargetFaceSize(cv::Size(_settings.value("Facetracking/FaceHSize",170).toInt(),
                                                  _settings.value("Facetracking/FaceVSize",226).toInt()));

    cv::CascadeClassifier _facecascade(a.applicationDirPath().append("/haarcascade_frontalface_alt2.xml").toUtf8().constData());
    if(_facecascade.empty()) {
        _qvideocapture.close();
        qWarning("  Can not load face classifier cascade! Abort...");
        return 6;
    } else {
        _qmultyfacetracker.setFaceClassifier(&_facecascade);
    }
    dlib::shape_predictor _dlibfaceshapepredictor;
    try {
        dlib::deserialize(a.applicationDirPath().append("/shape_predictor_5_face_landmarks.dat").toStdString()) >> _dlibfaceshapepredictor;
    }
    catch(...) {
        qWarning("  Can not load dlib's face shape predictor resources! Abort...");
        return 7;
    }
   _qmultyfacetracker.setDlibFaceShapePredictor(&_dlibfaceshapepredictor);
   _qmultyfacetracker.setFaceAlignmentMethod( FaceTracker::FaceShape );
   if(_visualization == false)
       _visualization = _settings.value("Miscellaneous/Visualization",false).toBool();
   _qmultyfacetracker.setVisualization(_visualization);
   qInfo("  Success");

    // Create face recognizer, later we also place them in the separate thread
    if(_identificationurl.isEmpty())
        _identificationurl = _settings.value("Facerecognition/ApiURL",QString()).toString();
    if(_identificationurl.isEmpty()) {
        qWarning("You have not provide identification resources location! Abort...");
        return 8;
    }
    QFaceRecognizer _qfacerecognizer(_identificationurl);
    // Let's create video locker
    QVideoLocker _qvideolocker;

    //MongoDB integration (through Eve REST full WEB interface)
    QMongoDBClient *_qmongodbclient;
    if((!_eveserverposturl.isEmpty()) && (!_mongodbaccesstoken.isEmpty())) {
        _qmongodbclient = new QMongoDBClient(&_qfacerecognizer);
        _qmongodbclient->setUrl(_eveserverposturl);
        _qmongodbclient->setToken(_mongodbaccesstoken);
        _qmongodbclient->setSpotid(_viewspotid);
        QObject::connect(&_qfacerecognizer, SIGNAL(labelPredicted(int,double,cv::String,cv::RotatedRect)), _qmongodbclient, SLOT(enrollRecognition(int,double,cv::String,cv::RotatedRect)));
    }

    // Slack integration
    QSlackClient *_qslackclient;
    QString _slackchannelid = _settings.value("Slack/ChannelID",QString()).toString();
    QString _slackbottoken = _settings.value("Slack/Bottoken",QString()).toString();
    if((!_slackchannelid.isEmpty()) && (!_slackbottoken.isEmpty())) {
        _qslackclient = new QSlackClient(&_qfacerecognizer);
        _qslackclient->setSlackchannelid(_slackchannelid);
        _qslackclient->setSlackbottoken(_slackbottoken);
        QObject::connect(&_qfacerecognizer, SIGNAL(labelPredicted(int,double,cv::String,cv::Mat)), _qslackclient, SLOT(enrollRecognition(int,double,cv::String,cv::Mat)));
    }

    // Oh, now let's make signals/slots connections
    QObject::connect(&_qvideocapture, SIGNAL(frameUpdated(cv::Mat)), &_qmultyfacetracker, SLOT(enrollImage(cv::Mat)), Qt::BlockingQueuedConnection);
    QObject::connect(&_qmultyfacetracker, SIGNAL(faceWithoutLabelFound(cv::Mat,cv::RotatedRect)), &_qvideolocker, SLOT(updateFrame(cv::Mat,cv::RotatedRect)));
    QObject::connect(&_qvideolocker, SIGNAL(frameUpdated(cv::Mat,cv::RotatedRect)), &_qfacerecognizer, SLOT(predict(cv::Mat,cv::RotatedRect)));
    QObject::connect(&_qfacerecognizer, SIGNAL(labelPredicted(int,double,cv::String,cv::RotatedRect)), &_qmultyfacetracker, SLOT(setLabelForTheFace(int,double,cv::String,cv::RotatedRect)));
    QObject::connect(&_qfacerecognizer, SIGNAL(labelPredicted(int,double,cv::String,cv::RotatedRect)), &_qvideolocker, SLOT(unlock()));

    qInfo(" Starting threads");
    // Let's organize threads
    QThread _qvideocapturethread; // a thread for the video capture
    _qvideocapture.moveToThread(&_qvideocapturethread);
    _qvideolocker.moveToThread(&_qvideocapturethread);
    QObject::connect(&_qvideocapturethread, SIGNAL(started()), &_qvideocapture, SLOT(init()));
    _qvideocapturethread.start();

    QThread _qfacetrackerthread; // a thred for the face tracker    
    _qmultyfacetracker.moveToThread(&_qfacetrackerthread);   
    _qfacetrackerthread.start(); 

    // Resume video capturing after timeout
    QTimer::singleShot(500, &_qvideocapture, SLOT(resume()));
    // Start to process events
    qInfo("  Success");
    return a.exec();
}

void logMessage(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    Q_UNUSED(context)
    if(p_logfile != nullptr) {

        QTextStream _logstream(p_logfile);
        switch (type) {
            case QtDebugMsg:
                _logstream << "[Debug]: ";
                break;
            case QtInfoMsg:
                _logstream << "[Info]: ";
                break;
            case QtWarningMsg:
                _logstream << "[Warning]: ";
                break;
            case QtCriticalMsg:
                _logstream << "[Critical]: ";
                break;
            case QtFatalMsg:
                _logstream << "[Fatal]: ";
                abort();
        }

        _logstream << QDateTime::currentDateTime().toString("(dd.MM.yyyy hh:mm:ss) ") << msg << "\n";

        // Check if logfile size exceeds size threshold 
        if(p_logfile->size() > 10E6) { // in bytes

            QString _filename = p_logfile->fileName();
            p_logfile->close(); // explicitly close logfile before remove
            QFile::remove(_filename); // delete from the hard drive
            delete p_logfile; // clean memory
            p_logfile = new QFile(_filename);

            if(!p_logfile->open(QIODevice::Append))
                delete p_logfile;
        }
    }
}

#include <QCoreApplication>
#include <QStringList>
#include <QDir>

#include "qupdatethread.h"

int main(int argc, char *argv[])
{
#ifdef Q_OS_WIN
    setlocale(LC_CTYPE,"Rus");
#endif

    char *dirname=nullptr, *apiurl=nullptr;
    unsigned int maxthreads = QThread::idealThreadCount();
    QUpdateThread::TaskType task = QUpdateThread::Unknown;
    bool asklabelinfo =  false;

    while(--argc > 0 && (*++argv)[0] == '-')
        switch(*++argv[0]) {
            case 't':
                maxthreads = QString(++argv[0]).toUInt();
                break;
            case 'i':
                dirname = ++argv[0];
                break;
            case 'a':
                apiurl = ++argv[0];
                break;
            case 'r':
                task = QUpdateThread::Remember;
                break;
            case 'd':
                task = QUpdateThread::Delete;
                break;
            case 'l':
                asklabelinfo = true;
                break;
            case 'h':
                qInfo("%s v.%s\n this application has been designed for automation of data uploading to openirt tools", APP_NAME, APP_VERSION);
                qInfo(" -a[apiurl]  - recognition web-server api url");                
                qInfo(" -i[dirname] - directory with labels and pictures in the subdirs");
                qInfo(" -t[uint]    - maximum number of worker threads");
                qInfo(" -r          - remember subdirname as labelinfo (or update if passed with -l)");
                qInfo(" -d          - delete subdirname as labelinfo (or delete all known if passed with -l)");
                qInfo(" -l          - use labels from server option");
                qInfo(" -h          - this help");
                qInfo("designed by %s in 2018", APP_DESIGNER);
                return 0;
        }
    if(task == QUpdateThread::Unknown) {
        qWarning("No task specified! Abort...");
        return 1;
    }
    if(apiurl == nullptr) {
        qWarning("Empty api url! Abort...");
        return 2;
    }
    if((task == QUpdateThread::Remember) && (dirname == nullptr)) {
        qWarning("Empty labels directory name! Abort...");
        return 3;
    }
    if(maxthreads == 0) {
        qWarning("You should specify non zero positive quantity of worker threads! Abort...");
        return 4;
    }

    QDir dir(dirname);
    if(dir.exists() == false) {
        qWarning("Empty labels directory name! Abort...");
        return 5;
    }

    QCoreApplication a(argc,argv);
    QStringList knownlabelinfo;
    if(asklabelinfo) {
        qInfo("Retrieving known labelinfo from the remote server:");
        knownlabelinfo = askLabelsInfoFrom(apiurl);
    }

    QList<QPair<QString,QString>> files;
    files.reserve(1024);

    if(asklabelinfo == false) {
        qInfo("Analyzing local directory:");
        qInfo("%s", dir.absolutePath().toUtf8().constData());
        QStringList lsubdirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        QStringList filefilters;
        filefilters << "*.jpg" << "*.jpeg" << "*.png";
        for(int i = 0; i < lsubdirs.size(); ++i) {
            QDir subdir(dir.absolutePath().append("/%1").arg(lsubdirs.at(i)));
            qInfo(" / %s", lsubdirs.at(i).toUtf8().constData());
            QStringList lfiles = subdir.entryList(filefilters, QDir::Files | QDir::NoDotAndDotDot);
            for(int j = 0; j < lfiles.size(); ++j) {
                qInfo("     / %s", lfiles.at(j).toUtf8().constData());
                files.push_back(qMakePair(lsubdirs.at(i),subdir.absoluteFilePath(lfiles.at(j))));
                if(task == QUpdateThread::Delete)
                    break;
            }
        }
    } else if(task == QUpdateThread::Remember) {
        qInfo("Analyzing local directory:");
        qInfo("%s", dir.absolutePath().toUtf8().constData());
        QStringList lsubdirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        QStringList filefilters;
        filefilters << "*.jpg" << "*.jpeg" << "*.png";
        for(int i = 0; i < lsubdirs.size(); ++i) {
            if( knownlabelinfo.contains(lsubdirs.at(i)) )
                continue;
            QDir subdir(dir.absolutePath().append("/%1").arg(lsubdirs.at(i)));
            qInfo(" / %s", lsubdirs.at(i).toUtf8().constData());
            QStringList lfiles = subdir.entryList(filefilters, QDir::Files | QDir::NoDotAndDotDot);
            for(int j = 0; j < lfiles.size(); ++j) {
                qInfo("     / %s", lfiles.at(j).toUtf8().constData());
                files.push_back(qMakePair(lsubdirs.at(i),subdir.absoluteFilePath(lfiles.at(j))));
            }
        }
    } else {
        for(int i = 0; i < knownlabelinfo.size(); ++i) {
            files.push_back(qMakePair(knownlabelinfo.at(i),QString()));
        }
    }

    unsigned int threadcounter = 0;
    QString url(apiurl);
    if(files.size() > 0) {
        qInfo("Processing tasks:");
        for(int i = 0; i < files.size(); ++i) {
            if(i != files.size()-1) {
                while(threadcounter >= maxthreads)
                    QCoreApplication::processEvents();
                QUpdateThread *thread = new QUpdateThread(&threadcounter,url,task,files.at(i).first,files.at(i).second);
                QObject::connect(thread,SIGNAL(finished()),thread,SLOT(deleteLater()));
                thread->start();
            } else { // last task should be executed differently, basically we need wait untill all other task will be accomplished
                while(threadcounter > 0)
                    QCoreApplication::processEvents();
                QUpdateThread *thread = new QUpdateThread(&threadcounter,url,task,files.at(i).first,files.at(i).second);
                QObject::connect(thread,SIGNAL(finished()),thread,SLOT(deleteLater()));
                QObject::connect(thread,SIGNAL(finished()),&a,SLOT(quit()));
                thread->start();
            }
        }
        return a.exec();
    }
    return 0;
}

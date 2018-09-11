#include <QCoreApplication>
#include "qupdatethread.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QUpdateThread *thread = new QUpdateThread();
    QObject::connect(thread,SIGNAL(finished()),thread,SLOT(deleteLater()));
    thread->start();

    return a.exec();
}

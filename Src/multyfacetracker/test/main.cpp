#include <string>
#include "opencv2/opencv.hpp"
#include "multyfacetracker.h"

#define FACE_CASCADE_FILENAME "/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"
#define EYE_CASCADE_FILENAME "/sources/data/haarcascades/haarcascade_eye.xml"

template<typename T>
std::string num2str(T value);

template<typename T>
T str2num(std::string str);

#ifdef ENABLE_OPENCL_CONTROLS
#include "opencv2/core/ocl.hpp"
#endif

void setupOpenCL();

int main(int argc, char *argv[])
{
    setupOpenCL();

    uint videoWidth = 640, videoHeight = 480, videodevID = 0;
    bool flipFlag = false;
    cv::String urlName, facecascadeName, eyecascadeName, outputFolderName;
    while(--argc > 0 && (*++argv)[0] == '-') {
        char m_option = *++argv[0];
        switch(m_option) {
        case 'v':
            videodevID = str2num<uint>( std::string(++(*argv)) );
            break;
        case 'f':
            flipFlag = true;
            break;
        case 'c':
            videoWidth = str2num<uint>( std::string(++(*argv)) );
            break;
        case 'r':
            videoHeight = str2num<uint>( std::string(++(*argv)) );
            break;
        case 'u':
            urlName = cv::String(++(*argv));
            break;
        case 'n':
            facecascadeName = cv::String(++(*argv));
            break;
        case 'e':
            eyecascadeName = cv::String(++(*argv));
            break;
        case 'o':
            outputFolderName = cv::String(++(*argv));
            break;
        case 'h':
            std::cout   << "Multyracker test"
                        << "\n -v - video capture device id (default 0)"
                        << "\n -f - enable vertical flip of input images (use if camera has been mounted head over heals)"
                        << "\n -c - set desired quantity of cols for video (default 640)"
                        << "\n -r - set desired quantity of rows for video (default 480)"
                        << "\n -u - desired URL to open as a video source"
                        << "\n -n - desired face detection cascade"
                        << "\n -e - desired eye detection cascade"
                        << "\n -o - output folder (if defined & exists all found faces will be saved, example: -oFaces/)"
                        << "\n -h - this help"
                        << "\ndesigned by Alex.A.Taranov, 2016";
            return 0;
        }
    }

    cv::CascadeClassifier faceClassifier, eyeClassifier;
    #ifdef OPENCV_DIRECTORY
        faceClassifier.load(std::string(OPENCV_DIRECTORY) + std::string(FACE_CASCADE_FILENAME));
        eyeClassifier.load(std::string(OPENCV_DIRECTORY) + std::string(EYE_CASCADE_FILENAME));
    #else
        faceClassifier.load(facecascadeName);
        eyeClassifier.load(eyecascadeName);
    #endif

    if(faceClassifier.empty() || eyeClassifier.empty()) {
        std::cout << "\nHaar cascades have not been loaded, check paths to files. Abort...";
        return -1;
    }
	
    MultyFaceTracker tracker;
    tracker.setFaceClassifier(&faceClassifier);
    tracker.setEyeClassifier(&eyeClassifier);

    cv::VideoCapture capture;
    if(urlName.empty() ? capture.open(0) : capture.open(urlName)) {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);
        cv::Mat frame;
        cv::RotatedRect rRect;
        cv::Point2f vertices[4];
        int symbol = 0;
        uint facesFound = 0;
        double frametime = 0.0;
        int64 timeMark = cv::getTickCount();
        while(capture.read(frame)) {
            if(!frame.empty()) {
				if(flipFlag)
                    cv::flip(frame,frame,0);
			
                std::vector<cv::Mat> v_facesImages = tracker.getResizedFaceImages(frame, cv::Size(128,156));
                for(uint i = 0; i < v_facesImages.size(); i++) {
                    cv::imshow(std::string("ID ")+std::to_string(i), v_facesImages[i]);
                    if(!outputFolderName.empty()) {
                        cv::imwrite(std::string(outputFolderName.c_str()) + std::string("Face_") + num2str(facesFound++) + std::string(".jpg"), v_facesImages[i]);
                    }
                }


                std::vector<cv::RotatedRect> v_faces = tracker.getRotatedRects();
                for(uint j = 0; j < v_faces.size(); j++) {
                    rRect = v_faces[j];
                    rRect.points(vertices);
                    cv::Scalar color = cv::Scalar(j * 255.0 / v_faces.size(), 0, 255 - j * 255.0 / v_faces.size());
                    for(uchar i = 0; i < 4; i++)
                        cv::line(frame, vertices[i], vertices[(i+1)%4], color,1, CV_AA);
                    cv::putText(frame, std::to_string(j), vertices[0] - cv::Point2f(-5,5), CV_FONT_HERSHEY_SIMPLEX, 0.5, color, 1, CV_AA);
                }


                cv::String _timestr = std::to_string(frametime) + std::string(" ms");

                cv::putText(frame, _timestr, cv::Point(10, frame.rows - 10),
                            CV_FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0,0,0),1, CV_AA);
                cv::putText(frame, _timestr, cv::Point(9, frame.rows - 11),
                            CV_FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255,255,255),1, CV_AA);

                cv::imshow("Test of multyface traacker", frame);

            }
            symbol = cv::waitKey(1);
            if(symbol == 27)
                break;
            frametime = 1000.0 * (cv::getTickCount() - timeMark) / cv::getTickFrequency();
            timeMark = cv::getTickCount();
        }
        capture.release();
    }
    return 0;
}

void setupOpenCL()
{
    std::cout << "--------------------------------------------------" << std::endl;
    #ifdef ENABLE_OPENCL_CONTROLS
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU))
        std::cout << "Failed creating the context...\n";
    else
        std::cout << context.ndevices() << " GPU devices are detected" ;
    for (int i = 0; i < context.ndevices(); i++)    {
        cv::ocl::Device device = context.device(i);
        std::cout << "\n\nName              : " << device.name();
        std::cout << "\nAvailable           : " << device.available();
        std::cout << "\nImageSupport        : " << device.imageSupport();
        std::cout << "\nOpenCL_C_Version    : " << device.OpenCL_C_Version();
    }
    // Uncomment next string to disable opencl backend of the opencv's
    //cv::ocl::setUseOpenCL(false);
    std::cout << (cv::ocl::useOpenCL() ? "\nOpencv will use Opencl\n" : "\nOpencv will NOT use Opencl\n");
    #else
    std::cout << "Define ENABLE_OPENCL_CONTROLS to get acces to the opencl controls" << std::endl;
    #endif
    std::cout << "--------------------------------------------------" << std::endl << std::endl;
}

template<typename T>
std::string num2str(T value)
{
    std::ostringstream os;
    os << value;
    return os.str();
}

template<typename T>
T str2num(std::string str)
{
    std::istringstream ss(str);
    T result;
    return (ss >> result ? result : 0);
}




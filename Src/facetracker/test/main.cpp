#include "opencv2/opencv.hpp"
#include "facetracker.h"
#include <istream>
#include <sstream>

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
            std::cout   << "Facetracker test"
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

    FaceTracker tracker(4,FaceTracker::NoAlign);
    tracker.setFaceClassifier(&faceClassifier);
    tracker.setEyeClassifier(&eyeClassifier);

    cv::VideoCapture capture;
    if(urlName.empty() ? capture.open(videodevID) : capture.open(urlName)) {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);
        cv::Mat frame;
        cv::Mat face;
        int symbol = 0;
		uint facesFound = 0;
        double frametime = 0.0;
        int64 timeMark = cv::getTickCount();
        while(capture.read(frame)) {
            if(flipFlag)
                cv::flip(frame,frame,0);
            face = tracker.getResizedFaceImage(frame,cv::Size(72,96));
            if(!face.empty()) {
                cv::imshow("Face", face);
                if(!outputFolderName.empty()) {
                    cv::imwrite(std::string(outputFolderName.c_str()) + std::string("/Face_") + num2str(facesFound++) + std::string(".png"), face);
                }
            }

            cv::RotatedRect rRect = tracker.getFaceRotatedRect();
            if(rRect.size.area() > 0) {
                cv::Point2f vertices[4];
                rRect.points(vertices);
                for(uchar i = 0; i < 4; i++)
                    cv::line(frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0),1, CV_AA);
                cv::circle(frame, rRect.center, 2, cv::Scalar(0,0,255), 1, CV_AA);
            }

            cv::putText(frame, num2str(frametime) + std::string(" ms"),
                        cv::Point(10, frame.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0,0,0),1, CV_AA);
			cv::putText(frame, num2str(frametime) + std::string(" ms"),
                        cv::Point(8, frame.rows - 12), CV_FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255,255,255),1, CV_AA);

            cv::putText(frame, std::string("Use 'f' to flip image and 'c' to adjust camera"),
                        cv::Point(12, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0,0,0),1, CV_AA);
            cv::putText(frame, std::string("Use 'f' to flip image and 'c' to adjust camera"),
                        cv::Point(10, 18), CV_FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255,255,255),1, CV_AA);
						
            cv::namedWindow("Test of the face tracker");
            cv::imshow("Test of the face tracker", frame);

            symbol = cv::waitKey(1);
            if(symbol == 27)
                break;
            else switch(symbol) {
                case 'f':
                    flipFlag = !flipFlag;
                    break;
                case 'c':
                    capture.set(CV_CAP_PROP_SETTINGS,0.0);
                    break;
            }
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
        std::vector<cv::ocl::PlatformInfo> platforms;
        cv::ocl::getPlatfomsInfo(platforms);
        std::cout << "Opencl platforms on this machine:" << std::endl;
        for (size_t i = 0; i < platforms.size(); i++) {
            //Access to Platform
            const cv::ocl::PlatformInfo* platform = &platforms[i];
            //Platform Name
            std::cout   << "\n\tPlatform # "<< i << " **********************\n"
                        << "\tName: " << platform->name() << std::endl
                        << "\tVendor: " << platform->vendor() << std::endl
                        << "\tVersion: " << platform->version() << std::endl
                        << "\tDevice Number: " << platform->deviceNumber() << std::endl;
        }

        /*std::cout << "\n--------------\nOpencl create context status: ";
        cv::ocl::Context context;
        if (!context.create(ocl::Device::TYPE_GPU))
            std::cout << "FAIL\n";
        else
            std::cout << "SUCCESS\n" ;

        std::cout << "\nOpencl devices on this machine:" << std::endl;
        for (int i = 0; i < context.ndevices(); i++)    {
            std::cout << "\n\tDevice # " << i;
            std::cout << "\n\tName              : " << context.device(i).name();
            std::cout << "\n\tVendor           : " << context.device(i).vendorName();
            std::cout << "\n\tVersion        : " << context.device(i).version();
            std::cout << "\n\tOpenCL_C_Version    : " << context.device(i).OpenCL_C_Version();
        }*/

        // Uncomment next string to disable opencl backend of the opencv's
        //cv::ocl::setUseOpenCL(false);
        std::cout << "\n--------------------------------------------------\n";
        std::cout << (cv::ocl::useOpenCL() ? "Opencl enabled\n" : "\nOpencl disabled\n");
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



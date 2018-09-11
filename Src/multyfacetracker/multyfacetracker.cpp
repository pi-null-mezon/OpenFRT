#include "multyfacetracker.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

MultyFaceTracker::MultyFaceTracker(uint maxfaces, uchar length, FaceTracker::AlignMethod method)
{
    m_historylength = length;
    m_alignmethod = method;
    setMaxFaces(maxfaces);
}

MultyFaceTracker::~MultyFaceTracker()
{
    for(uint i = 0; i < v_facetrackers.size(); i++)
        delete v_facetrackers[i];
}

std::vector<cv::RotatedRect> MultyFaceTracker::searchFaces(const cv::Mat &Img)
{
    cv::Mat ImgCopy = Img.clone();
    std::vector<cv::RotatedRect> v_rects;
    for(uint i = 0; i < v_facetrackers.size(); i++) {
		
		// multiple facetrackers are used because each one of them stores face location history
        FaceTracker *pt_faceTracker = v_facetrackers[i];
        cv::RotatedRect rRect = pt_faceTracker->searchFace(ImgCopy);

        if(rRect.size.area() > 0) {
            v_rects.push_back(rRect);
            if(v_facetrackers.size() > 1) {
                cv::rectangle(ImgCopy,rRect.boundingRect(), cv::Scalar(0,0,0),-1);
                //cv::ellipse(ImgCopy,rRect,cv::Scalar(0,127,0),-1);
            }
        } else {
            for(uint j = i; j < v_facetrackers.size(); ++j) {
                FaceTracker *_ptracker = v_facetrackers[j];
                _ptracker->clearMetaData();
                _ptracker->resetHistory();
            }
            break;
        }
    }
    m_facesFound = (uint)v_rects.size();
    return v_rects;
}

bool MultyFaceTracker::setEyeClassifier(cv::CascadeClassifier *pointer)
{
    for(uint i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *pt_faceTracker = v_facetrackers[i];
        if(!pt_faceTracker->setEyeClassifier(pointer))
            return false;
    }
    return true;
}

void MultyFaceTracker::setDlibFaceShapePredictor(dlib::shape_predictor *pointer)
{
    for(uint i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *pt_faceTracker = v_facetrackers[i];
        pt_faceTracker->setFaceShapePredictor(pointer);
    }
}

bool MultyFaceTracker::setFaceClassifier(cv::CascadeClassifier *pointer)
{
    for(uint i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *pt_faceTracker = v_facetrackers[i];
        if(!pt_faceTracker->setFaceClassifier(pointer))
            return false;
    }
    return true;
}

std::vector<cv::Mat> MultyFaceTracker::getResizedFaceImages(const cv::Mat &Img, const cv::Size size)
{
    cv::Mat ImgCopy = Img.clone();

    std::vector<cv::Mat> v_facesImages;
    for(uint i = 0; i < v_facetrackers.size(); i++) {

        FaceTracker *pt_faceTracker = v_facetrackers[i];
        cv::Mat faceImage = pt_faceTracker->getResizedFaceImage(ImgCopy, size);
        if(!faceImage.empty()) {
            v_facesImages.push_back(faceImage);
            if(v_facetrackers.size() > 1) {
                cv::rectangle(ImgCopy,pt_faceTracker->getFaceRotatedRect().boundingRect(), cv::Scalar(0,0,0),-1);
                //cv::ellipse(ImgCopy,pt_faceTracker->getFaceRotatedRect(),cv::Scalar(0,127,0),-1);
            }
        } else {
            //std::cout << "-----------" << std::endl;
            for(uint j = i; j < v_facetrackers.size(); ++j) {
                FaceTracker *_ptracker = v_facetrackers[j];
                _ptracker->clearMetaData();
                _ptracker->resetHistory();
                //std::cout << j << ") reset" << std::endl;
            }
            break;
        }
    }
    m_facesFound = (uint)v_facesImages.size();
    return v_facesImages;
}

std::vector<cv::RotatedRect> MultyFaceTracker::getRotatedRects() const
{
    std::vector<cv::RotatedRect> v_rRects;
    for(uint i = 0; i < m_facesFound; i++) {
        FaceTracker *tracker = v_facetrackers[i];
        v_rRects.push_back(tracker->getFaceRotatedRect());
    }
    return v_rRects;
}

void MultyFaceTracker::setMaxFaces(uint maxfaces)
{
    for(uint i = 0; i < v_facetrackers.size(); i++)
        delete v_facetrackers[i];
    v_facetrackers.clear();

    m_facesFound = 0;
    for(uint i = 0; i < maxfaces; i++)
        v_facetrackers.push_back(new FaceTracker(m_historylength, m_alignmethod));
}

size_t MultyFaceTracker::getMaxFaces() const
{
    return v_facetrackers.size();
}

void MultyFaceTracker::setFaceRectPortions(float xPortion, float yPortion)
{
    for(size_t i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *tracker = v_facetrackers[i];
        tracker->setFaceRectPortions(xPortion,yPortion);
    }
}

void MultyFaceTracker::setFaceRectShifts(float _xShift, float _yShift)
{
    for(size_t i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *tracker = v_facetrackers[i];
        tracker->setFaceRectShifts(_xShift,_yShift);
    }
}

const FaceTracker *MultyFaceTracker::at(int i)
{
    return v_facetrackers[i];
}

FaceTracker *MultyFaceTracker::operator[](int i)
{
    return v_facetrackers[i];
}

void MultyFaceTracker::setFaceAlignMethod(FaceTracker::AlignMethod _method)
{
    m_alignmethod = _method;
    for(size_t i = 0; i < v_facetrackers.size(); i++) {
        FaceTracker *tracker = v_facetrackers[i];
        tracker->setFaceAlignMethod(_method);
    }
}

FaceTracker::AlignMethod MultyFaceTracker::getFaceAlignMethod() const
{
    return m_alignmethod;
}






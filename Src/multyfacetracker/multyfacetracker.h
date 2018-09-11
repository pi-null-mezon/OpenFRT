#ifndef MULTYFACETRACKER_H
#define MULTYFACETRACKER_H

#include "facetracker.h"
#include <vector>

/**
 * @brief The MultyFaceTracker class was designed for multi face tracking
 */
class MultyFaceTracker
{
public:
    /**
     * @brief Class constructor
     * @param maxfaces - maximum of simultaneously tracked faces
     * @param length - value of position history length for the facetrakers
     * @param method - type of alignment strategy
     */
    MultyFaceTracker(uint maxfaces = 5, uchar length = 4, FaceTracker::AlignMethod method = FaceTracker::AlignMethod::Eyes);
    ~MultyFaceTracker();
    /**
     * @brief runs iterative procedure of face search and track
     * @param Img - input image
     * @return vector of faces position on Img in cv::RotatedRect format
     */
    std::vector<cv::RotatedRect> searchFaces(const cv::Mat &Img);
    /**
     * @brief runs iterative procedure of face search and track
     * @param Img - input image
     * @param size - output images size
     * @return vector of resized faces images
     */
    std::vector<cv::Mat> getResizedFaceImages(const cv::Mat &Img, const cv::Size size);
    /**
     * @brief loadFaceClassifier
     * @param pointer - pointer to an object instance
     * @return has classifier been loaded or not
     */
    bool setFaceClassifier(cv::CascadeClassifier *pointer);
    /**
     * @brief loadEyeClassifier
     * @param pointer - pointer to an object instance
     * @return has classifier been loaded or not
     */
    bool setEyeClassifier(cv::CascadeClassifier *pointer);
    /**
     * @brief setDlibFaceShapePredictor
     * @param pointer
     */
    void setDlibFaceShapePredictor(dlib::shape_predictor *pointer);
    /**
     * @brief getRotatedRects
     * @return vector of faces position on Img in cv::RotatedRect format
     * @note should be called after searchFaces(...) or getResizedFaceImages(...)
     */
    std::vector<cv::RotatedRect> getRotatedRects() const;
    /**
     * @brief setMaxFaces
     * @param maxfaces - maximum of simultaneously tracked faces
     */
    void setMaxFaces(uint maxfaces);
    /**
     * @brief getMaxFaces
     * @return size of the facetrackers vector
     */
    size_t getMaxFaces() const;
    /**
     * @brief calls setFaceRectPortions for the all FaceRecognizers
     * @param xPortion - self explained
     * @param yPortion - self explained
     */
    void setFaceRectPortions(float xPortion, float yPortion);
    /**
     * @brief calls setFaceRectShifts for the all FaceRecognizers
     * @param _xShift - shift in portion of face rect width
     * @param _yShift - shift in portion of face rect height
     */
    void setFaceRectShifts(float _xShift, float _yShift);
    /**
     * @param i - index of the tracker, it should be greater than -1 and less than getMaxFaces()
     * @return pointer to the particular facetracker
     */
    const FaceTracker *at(int i);
    /**
     * @param i - index of the tracker, it should be greater than -1 and less than getMaxFaces()
     * @return pointer to the particular facetracker
     */
    FaceTracker *operator[](int i);
    /**
     * @brief setFaceAlignMethod - set face align method for the each one of facetrackers
     * @param _method - self explained
     */
    void setFaceAlignMethod(FaceTracker::AlignMethod _method);
    /**
     * @brief getFaceAlignMethod - self explained
     * @return face align method
     */
    FaceTracker::AlignMethod getFaceAlignMethod() const;

private:
    std::vector<FaceTracker *> v_facetrackers;
    uint m_facesFound;
    uint m_historylength;
    FaceTracker::AlignMethod m_alignmethod;
};

#endif // MULTYFACETRACKER_H

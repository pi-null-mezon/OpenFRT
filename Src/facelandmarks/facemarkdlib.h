#ifndef FACEMARKDLIB_H
#define FACEMARKDLIB_H

#include <opencv2/core.hpp>
#include <opencv2/face/facemark.hpp>

#include <dlib/image_processing/shape_predictor.h>

namespace cv { namespace face {

/**
 * @brief The FacemarkDlib class is a wrap for the dlib's 68 (and 5 also) facial points detector
 */
class FacemarkDlib : public Facemark {

public:
    FacemarkDlib();

    void loadModel(String model);

    bool fit( InputArray image,
              InputArray faces,
              OutputArrayOfArrays landmarks);

private:
    mutable dlib::shape_predictor shapepredictor;
};

Ptr<Facemark> createFacemarkDlib();

}}




#endif // FACEMARKDLIB_H

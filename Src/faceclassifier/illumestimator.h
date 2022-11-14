#ifndef ILLUMESTIMATOR_H
#define ILLUMESTIMATOR_H

#include "faceclassifier.h"

namespace cv { namespace ofrt {

class IllumEstimator : public FaceClassifier
{
public:
    IllumEstimator(const std::string &modelfilename);

    std::vector<float> process(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks, bool fast) override;

    static Ptr<FaceClassifier> createClassifier(const std::string &modelfilename="");
};

}}

#endif // ILLUMESTIMATOR_H

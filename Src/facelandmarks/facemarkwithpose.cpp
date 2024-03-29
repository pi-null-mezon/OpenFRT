#include "facemarkwithpose.h"

#include <opencv2/imgproc.hpp>

namespace cv { namespace ofrt {

FacemarkWithPose::FacemarkWithPose(const String &modelfilename) :
    Facemark(),
    isize(cv::Size(100,100))
{
    net = cv::dnn::readNet(modelfilename);
    CV_Assert(!net.empty());
#ifdef FORCE_OPENCV_DNN_TO_USE_CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
    // Now read names of outbut layers
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<String> layersNames = net.getLayerNames();
    output_names.resize(outLayers.size());
    for(size_t i = 0; i < outLayers.size(); ++i)
        output_names[i] = layersNames[static_cast<size_t>(outLayers[i]) - 1];
}

bool FacemarkWithPose::fit(const cv::Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks) const
{
    poses.resize(0);
    return fit(image,faces,landmarks,poses);
}

std::vector<std::vector<float> > FacemarkWithPose::last_pose() const
{
    return poses;
}

bool FacemarkWithPose::fit(const Mat &image, const std::vector<Rect> &faces, std::vector<std::vector<Point2f>> &landmarks, std::vector<std::vector<float>> &angles) const
{   
    if(image.empty() || (faces.size() < 1))
        return false;
    landmarks.reserve(faces.size());
    angles.reserve(faces.size());
    const cv::Rect2f frame(0,0,image.cols,image.rows);
    for(const auto &rect : faces) {
        cv::Rect2f _rect = prepareRect(rect,frame,1.9);
        cv::Rect2f _roirect;
        cv::Mat blob;
        // trained with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        cv::dnn::blobFromImage(cropInsideFromCenterAndResize(image(_rect & frame),isize,_roirect),
                               blob,1.0/57.12,cv::Size(),cv::Scalar(123.675, 116.28, 103.53),true,false);
        std::vector<Mat> output_blobs;
        net.setInput(blob);
        net.forward(output_blobs, output_names);
        static int angles_idx = output_blobs[0].total() == 3 ? 0 : 1;
        static int landmarks_idx = angles_idx == 0 ? 1 : 0;

        // face landmarks
        float* data = reinterpret_cast<float*>(output_blobs[landmarks_idx].data);
        std::vector<cv::Point2f> _points;
        _points.reserve(output_blobs[landmarks_idx].total() / 2);
        for(size_t i = 0; i < output_blobs[landmarks_idx].total() / 2; ++i)
            _points.push_back(cv::Point2f((0.5f + data[2*i]) * _roirect.width + _rect.x + _roirect.x,
                                          (0.5f + data[2*i+1]) * _roirect.height + _rect.y + _roirect.y));
        landmarks.push_back(std::move(_points));
        data = reinterpret_cast<float*>(output_blobs[angles_idx].data);
        std::vector<float> tmp_angles(3,0);
        tmp_angles[0] = -90.0f * data[1]; // - yaw
        tmp_angles[1] = -90.0f * data[0]; // - pitch
        tmp_angles[2] = -90.0f * data[2]; // - roll
        angles.push_back(std::move(tmp_angles));
    }
    return true;
}

Ptr<Facemark> FacemarkWithPose::create(const String &modelfilename)
{
    return makePtr<FacemarkWithPose>(modelfilename);
}

}}

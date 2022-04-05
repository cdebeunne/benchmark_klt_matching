#include "Frame.hpp"

std::vector<cv::KeyPoint> Frame::getKeyPointsVector() const{
    std::vector<cv::KeyPoint> cv_kps;
    for( const auto &kp : _mapkps) {
        cv_kps.push_back(kp.second._cvKeyPoint);
    }
    return cv_kps;
}

cv::Mat Frame::getDescriptors() const{
    cv::Mat descs;
    for( const auto &kp : _mapkps) {
        descs.push_back(kp.second._desc);
    }
    return descs;
}

std::vector<cv::Point2f> Frame::getP2fVector() const{
    std::vector<cv::Point2f> p2fs;
    for( const auto &kp : _mapkps) {
        p2fs.push_back(kp.second._cvKeyPoint.pt);
    }
    return p2fs;
}

cv::Mat Frame::getCvImage() const{
    return _cvimg;
}

void Frame::addKeyPoint(KeyPoint kp){

    if(_mapkps.count(kp._id)){
        std::cout << "Warning, replacing a feature with the same ID" << std::endl;
    }
    _mapkps.emplace(kp._id, kp);

}

void Frame::removeKeyPoint(KeyPoint kp){
    _mapkps.erase(kp._id);
}


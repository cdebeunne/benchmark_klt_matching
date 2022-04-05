#include "Frame.hpp"

std::vector<KeyPoint> Frame::getKeyPointsVector() const{
    std::vector<KeyPoint> kps;
    for( const auto &kp : _mapkps) {
        kps.push_back(kp.second);
    }
    return kps;
}

std::vector<cv::KeyPoint> Frame::getCvKeyPointsVector() const{
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

std::vector<size_t> Frame::getKeyPointIndices() const{
    std::vector<size_t> kpindices;
    for( const auto &kp : _mapkps) {
        kpindices.push_back(kp.first);
    }
    return kpindices;
}

KeyPoint Frame::getKeyPointIdx(size_t idx) const{
    return _mapkps.at(idx);
}

cv::Mat Frame::getCvImage() const{
    return _cvimg;
}

void Frame::addKeyPoint(KeyPoint kp){

    if(_mapkps.count(kp._id)){
        std::cout << "Warning, replacing a feature with the same ID" << std::endl;
    }
    _mapkps.emplace(kp._id, kp);
    _maxidx++;

}

void Frame::addKeyPoint(cv::KeyPoint cvkp){
    KeyPoint kp;
    kp._id = _maxidx;
    kp._cvKeyPoint = cvkp;
    _mapkps.emplace(kp._id, kp);
    _maxidx++;
}

void Frame::addKeyPoint(cv::KeyPoint cvkp, cv::Mat desc){
    KeyPoint kp;
    kp._id = _maxidx;
    kp._desc = desc;
    kp._cvKeyPoint = cvkp;
    _mapkps.emplace(kp._id, kp);
    _maxidx++;
}


void Frame::removeKeyPoint(KeyPoint kp){
    _mapkps.erase(kp._id);
}

void Frame::removeKeyPointIdx(size_t idx){
    _mapkps.erase(idx);
}

void Frame::setImg(cv::Mat img){
    _cvimg = img;
}

void Frame::reset(){
    _mapkps.clear();
}


#ifndef FEATUREDETECTOR_H
#define FEATUREDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Frame.hpp>

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

cv::Ptr<cv::FeatureDetector> factoryDetector(Config config) {
    if (config.detector == "fast"){
        return cv::FastFeatureDetector::create(config.threshold_fast);
    }
    else if (config.detector == "orb"){
        int nb_orb_detected_local = config.nb_orb_detected / (config.nrows*config.ncols);
        return cv::ORB::create(nb_orb_detected_local,
                               config.scale_factor,
                               config.nlevels_pyramids,
                               31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    }
    else {
        throw std::invalid_argument("config.detector value not suppported: "+config.detector);
    }
}

void parallelDetect(Frame &f, cv::Ptr<cv::FeatureDetector> detector, int rows, int cols) {

    cv::Mat img = f.getCvImage();
    std::vector<cv::KeyPoint> keypoints;
    int w = img.cols / cols;
    int h = img.rows / rows;
    cv::Size s = img.size();
    int border = 5;


    // Define local detection function
    std::mutex mtx;
    auto detectComputeSmall = [w, h, cols, rows, detector, img, s, border, &mtx, &keypoints](
            int row, int col) {

        int x = std::max((int)col * w - border, 0);
        int y = std::max((int)row * h - border, 0);

        int width, height;

        if (row + 1 == rows) {
            height = s.height - y;
        } else {
            height = h + 2 * border;
        }

        if (col + 1 == cols) {
            width = s.width - x;
        } else {
            width = w + 2 * border;
        }

        cv::Rect RectangleToSelect(x, y, width, height);
        cv::Mat I = img(RectangleToSelect);

        std::vector<cv::KeyPoint> keypoints_local;
        cv::Mat descriptors_local;

        // call OpenCV detect function
        detector->detect(I, keypoints_local);


        // Add local detection to the full detection
        {
            mtx.lock();
            for (uint i = 0; i < keypoints_local.size(); i++) {
                keypoints_local.at(i).pt += cv::Point2f(x, y);
                keypoints.push_back(keypoints_local.at(i));
            }
            mtx.unlock();
        }
    };

    // Launch on different threads the local detections
    std::vector<std::thread> threads;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            threads.push_back(std::thread(detectComputeSmall, r, c));
        }
    }
    for (auto &th : threads) {
        th.join();
    }

    for (size_t i = 0; i<keypoints.size(); i++){
        f.addKeyPoint(keypoints.at(i));
    }

}

void parallelDetectAndCompute(Frame &f, cv::Ptr<cv::FeatureDetector> detector, int cols, int rows) {

    cv::Mat img = f.getCvImage();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    int w = img.cols / cols;
    int h = img.rows / rows;
    cv::Size s = img.size();
    int border = 5;


    // Define local detection function
    std::mutex mtx;
    auto detectComputeSmall = [w, h, cols, rows, detector, img, s, border, &mtx, &keypoints, &descriptors](
            int row, int col) {

        int x = std::max((int)col * w - border, 0);
        int y = std::max((int)row * h - border, 0);

        int width, height;

        if (row + 1 == rows) {
            height = s.height - y;
        } else {
            height = h + 2 * border;
        }

        if (col + 1 == cols) {
            width = s.width - x;
        } else {
            width = w + 2 * border;
        }

        cv::Rect RectangleToSelect(x, y, width, height);
        cv::Mat I = img(RectangleToSelect);

        std::vector<cv::KeyPoint> keypoints_local;
        cv::Mat descriptors_local;

        // call OpenCV detectAndCompute function
        detector->detectAndCompute(I, cv::Mat(), keypoints_local, descriptors_local);


        // Add local detection to the full detection
        {
            mtx.lock();
            for (uint i = 0; i < keypoints_local.size(); i++) {
                keypoints_local.at(i).pt += cv::Point2f(x, y);
                keypoints.push_back(keypoints_local.at(i));
            }
            descriptors.push_back(descriptors_local);
            mtx.unlock();
        }
    };

    // Launch on different threads the local detections
    std::vector<std::thread> threads;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            threads.push_back(std::thread(detectComputeSmall, r, c));
        }
    }
    for (auto &th : threads) {
        th.join();
    }

    for (size_t i = 0; i<keypoints.size(); i++){
        cv::Mat desc = descriptors.row(i);
        cv::KeyPoint kp = keypoints.at(i);
        f.addKeyPoint(kp, desc);
    }

}

#endif
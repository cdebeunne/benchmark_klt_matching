#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include "Timer.hpp"
#include "ConfigReader.hpp"
#include "FeatureTracker.hpp"
#include "FeatureMatcher.hpp"
#include "ImgLoader.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <filesystem>


int main(int argc, char** argv){

    // initialize K
    Eigen::Matrix3d K;
    K << 458.654, 0, 367.215,
         0, 457.296, 248.375,
         0, 0, 1;
    double focal_length = K(0);
    cv::Point2d principal_pt(K(0,2), K(1,2));

    // Load config
    Config config = readParameterFile(std::filesystem::current_path().string()+"/../"+"param.yaml");

    // path and loader of the EUROC sequence
    std::vector<std::string> img_list = EUROC_img_loader(config.dataset_path);
    cv::Mat img_curr, img_prev;

    // Initialize the detector
    cv::Ptr<cv::FeatureDetector> detector;
    if (config.detector == "fast"){
        detector = cv::FastFeatureDetector::create(config.threshold_fast);
    }
    else if (config.detector == "orb"){
        detector = cv::ORB::create(config.npoints,
                                            config.scale_factor,
                                            config.nlevels_pyramids,
                                            31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    }
    cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create(config.npoints,
                                            config.scale_factor,
                                            config.nlevels_pyramids,
                                            31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    std::vector<cv::KeyPoint> keypoints_curr, keypoints_prev;
    cv::Mat descriptors_curr, descriptors_prev;

    // Stores results in .csv
    std::fstream results;
    std::string results_path = "results.csv";
    results.open(results_path, std::fstream::out);
    results << "iter, nTracks, \n";


    // Here we detect once and try to see how many tracks and matches follow
    // BEWARE you can't run both tracking and matching
    
    int counter = 0;
    std::string img_path;

    for (const auto & img_name : img_list){
        img_path = config.dataset_path + "/data/" + img_name;
        if (counter > config.nimages) break;

        if (counter == 0){
            img_prev = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            detector->detect(img_prev, keypoints_prev);
            descriptor->compute(img_prev, keypoints_prev, descriptors_prev);
            results << 0 << "," << keypoints_prev.size() << ",\n";
            counter ++;
            continue;
        }

        img_curr = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        if (config.enable_tracker){
            // Create cv point list for tracking, we initialize optical flow with previous keypoints
            std::vector<cv::Point2f> p2f_curr, p2f_prev;
            for (auto & keypoint : keypoints_prev){
                p2f_prev.push_back(keypoint.pt);
                p2f_curr.push_back(keypoint.pt);
            }

            int ntracked_features = track(img_prev, img_curr, p2f_prev, p2f_curr,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
            // We need at least 5 kp to compute Essential
            if (p2f_curr.size() < 5) break;

            // Filtering with Essential Matrix
            cv::Mat cvMask, E;
            E = cv::findEssentialMat(p2f_prev, p2f_curr, focal_length, principal_pt, cv::RANSAC,
                                 0.99, 1.0, cvMask);
            std::vector<cv::Point2f> p2f_curr_filtered;

            for (size_t k=0; k<p2f_prev.size(); k++){
                if (cvMask.at<bool>(k) == 1){
                    p2f_curr_filtered.push_back(p2f_curr.at(k));
                }
            }

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";

            // Keypoint current are now the one that were tracked
            keypoints_prev.clear();
            for (size_t i = 0; i < p2f_curr_filtered.size(); i++){
                cv::KeyPoint keypoint;
                keypoint.pt = p2f_curr_filtered.at(i);
                keypoints_prev.push_back(keypoint);
            }
        }

        if (config.enable_matcher){
            // Detection
            detector->detect(img_curr, keypoints_curr);

            // Description
            descriptor->compute(img_curr, keypoints_curr, descriptors_curr);

            int nmatched_features = match(keypoints_prev, keypoints_curr, descriptors_prev, descriptors_curr,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);


            if (config.debug){
                std::cout << "Number of matches" << std::endl;
                std::cout << nmatched_features << std::endl;

                // Image Display
                cv::Mat img_matches;
                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < keypoints_curr.size(); i++){
                    cv::DMatch match(i,i,1);
                    good_matches.push_back(match);
                }
                cv::drawMatches(img_prev, keypoints_prev, img_curr, keypoints_curr, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }

            // We need at least 5 kp to compute Essential
            if (keypoints_curr.size() < 5) break;

            // Filtering with Essential Matrix
            std::vector<cv::Point2f> p2f_curr, p2f_prev;
            for (size_t k=0; k<keypoints_prev.size(); k++){
                p2f_prev.push_back(keypoints_prev.at(k).pt);
                p2f_curr.push_back(keypoints_curr.at(k).pt);
            }
            cv::Mat cvMask, E;
            E = cv::findEssentialMat(p2f_prev, p2f_curr, focal_length, principal_pt, cv::RANSAC,
                                 0.99, 1.0, cvMask);

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";

            // Keypoint previouse are now the one that were matched
            keypoints_prev.clear();
            for (size_t i = 0; i < p2f_curr.size(); i++){
                if (cvMask.at<bool>(i) == 1){
                    cv::KeyPoint keypoint;
                    keypoint = keypoints_curr.at(i);
                    keypoints_prev.push_back(keypoint);
                }
            }
            descriptor->compute(img_curr, keypoints_prev, descriptors_prev);
        }

        // Set curr as previous (only for image this time)
        img_prev = img_curr;
        counter++;
    }

    results.close();
}
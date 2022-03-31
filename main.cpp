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

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <filesystem>

int main(int argc, char** argv){

    Timer timer;

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

    // Initialize pertinent data
    float avg_track = 0;
    float avg_match = 0;
    float dt_detect = 0;
    float dt_match = 0;
    float dt_track = 0;
    float inliers_match = 0;
    float inliers_track = 0;

    int counter = 0;
    std::string img_path;
    for (const auto & img_name : img_list){
        img_path = config.dataset_path + "/data/" + img_name;
        if (counter > config.nimages) break;

        // Matches variable for vizu
        std::vector<cv::DMatch> good_matches;

        if (counter == 0){
            img_prev = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            detector->detect(img_prev, keypoints_prev);
            descriptor->compute(img_prev, keypoints_prev, descriptors_prev);
            counter ++;
            continue;
        }

        img_curr = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        // Detection
        timer.start();
        detector->detect(img_curr, keypoints_curr);
        timer.stop();
        dt_detect += timer.elapsedSeconds();
        

        if (config.enable_tracker){
            // Create cv point list for tracking, we initialize optical flow with previous keypoints
            std::vector<cv::Point2f> p2f_curr, p2f_prev;
            for (auto & keypoint : keypoints_prev){
                p2f_prev.push_back(keypoint.pt);
                p2f_curr.push_back(keypoint.pt);
            }

            timer.start();
            int ntracked_features = track(img_prev, img_curr, p2f_prev, p2f_curr,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
            timer.stop();
            dt_track += timer.elapsedSeconds();
            avg_track += ntracked_features;

            // Check inliers with essential
            cv::Mat cvMask, E;
            E = cv::findEssentialMat(p2f_prev, p2f_curr, focal_length, principal_pt, cv::RANSAC,
                                 0.99, 1.0, cvMask);
            inliers_track += 100 * cv::countNonZero(cvMask)/p2f_prev.size();

            if (config.debug){
                std::cout << "Number of tracks" << std::endl;
                std::cout << ntracked_features << std::endl;

                // Image display for debug
                cv::Mat img_tracks;
                std::vector<cv::KeyPoint> keypoints_curr_flow, keypoints_prev_flow;
                for (size_t i = 0; i < p2f_curr.size(); i++){
                    cv::DMatch match(i,i,1);
                    good_matches.push_back(match);
                    cv::KeyPoint kp_curr_flow, kp_prev_flow;
                    kp_curr_flow.pt = p2f_curr.at(i);
                    kp_prev_flow.pt = p2f_prev.at(i);
                    keypoints_curr_flow.push_back(kp_curr_flow);
                    keypoints_prev_flow.push_back(kp_prev_flow);
                }
                cv::drawMatches(img_prev, keypoints_prev_flow, img_curr, keypoints_curr_flow, good_matches, img_tracks, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_tracks );
                cv::waitKey(0);
            }
        }

        if (config.enable_matcher){
            std::vector<cv::KeyPoint> keypoints_prev_match = keypoints_prev;
            cv::Mat descriptors_prev_match = descriptors_prev;

            // Compute descriptors and matching 
            timer.start();
            descriptor->compute(img_curr, keypoints_curr, descriptors_curr);
            std::vector<cv::KeyPoint> keypoints_curr_match = keypoints_curr; 
            cv::Mat descriptors_curr_match = descriptors_curr;

            int nmatched_features = match(keypoints_prev_match, keypoints_curr_match, descriptors_prev_match, descriptors_curr_match,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
            timer.stop();
            dt_match += timer.elapsedSeconds();
            avg_match += nmatched_features;

            // Check inliers with essential
            std::vector<cv::Point2f> p2f_curr, p2f_prev;
            for (size_t k=0; k<keypoints_prev_match.size(); k++){
                p2f_prev.push_back(keypoints_prev_match.at(k).pt);
                p2f_curr.push_back(keypoints_curr_match.at(k).pt);
            }
            cv::Mat cvMask, E;
            E = cv::findEssentialMat(p2f_prev, p2f_curr, focal_length, principal_pt, cv::RANSAC,
                                 0.99, 1.0, cvMask);
            inliers_match += 100 * cv::countNonZero(cvMask)/p2f_prev.size();


            if (config.debug){
                std::cout << "Number of matches" << std::endl;
                std::cout << nmatched_features << std::endl;

                // Image Display
                cv::Mat img_matches;
                good_matches.clear();
                for (size_t i = 0; i < keypoints_curr_match.size(); i++){
                    cv::DMatch match(i,i,1);
                    good_matches.push_back(match);
                }
                cv::drawMatches(img_prev, keypoints_prev_match, img_curr, keypoints_curr_match, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }
        }

        // Set curr as previous 
        img_prev = img_curr;
        keypoints_prev = keypoints_curr;
        descriptors_prev = descriptors_curr;
        counter++;
    }

    avg_match /= counter;
    avg_track /= counter;
    dt_detect /= counter;
    dt_match /= counter;
    dt_track /= counter;
    inliers_match /= counter;
    inliers_track /= counter;

    std::cout << "----- RECAP -----" << std::endl;
    std::cout << "Number of successfull tracks" << std::endl;
    std::cout << avg_track << std::endl;
    std::cout << "Inliers ratio tracking" << std::endl;
    std::cout << inliers_track << std::endl;
    std::cout << "Number of successfull matches" << std::endl;
    std::cout << avg_match << std::endl;
    std::cout << "Inliers ratio matching" << std::endl;
    std::cout << inliers_match << std::endl;
    std::cout << "Time to detect" << std::endl;
    std::cout << dt_detect << std::endl;
    std::cout << "Time to track" << std::endl;
    std::cout << dt_track << std::endl;
    std::cout << "Time to match" << std::endl;
    std::cout << dt_match << std::endl;


    return 0;

}

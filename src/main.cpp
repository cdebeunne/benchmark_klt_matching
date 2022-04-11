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
#include "Multiview.hpp"
#include "ImgLoader.hpp"
#include "FeatureDetector.hpp"
#include <Frame.hpp>

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
    Eigen::Matrix3f K;
    K << 458.654, 0, 367.215,
         0, 457.296, 248.375,
         0, 0, 1;

    // Load config
    Config config = readParameterFile(std::filesystem::current_path().string()+"/../"+"param.yaml");

    // path and loader of the EUROC sequence
    std::vector<std::string> img_list = EUROC_img_loader(config.dataset_path);
    cv::Mat img_inc, img_last;
    Frame frame_last, frame_inc;

    // Initialize the detector
    cv::Ptr<cv::FeatureDetector> detector;
    if (config.detector == "fast"){
        detector = cv::FastFeatureDetector::create(config.threshold_fast);
    }
    else if (config.detector == "orb"){
        int npoints_local = config.npoints / (config.nrows*config.ncols);
        detector = cv::ORB::create(npoints_local,
                                   config.scale_factor,
                                   config.nlevels_pyramids,
                                   31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    }
    cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create(config.npoints,
                                            config.scale_factor,
                                            config.nlevels_pyramids,
                                            31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);

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
            img_last = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            frame_last.setImg(img_last);
            parallelDetectAndCompute(frame_last, detector, config.nrows, config.ncols);
            counter ++;
            continue;
        }
        img_inc = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        if (config.enable_tracker){
            frame_inc.reset();
            frame_inc.setImg(img_inc);
            timer.start();
            std::map<int, int> last_map_inc;
            int ntracked_features = track(frame_last, frame_inc, last_map_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
            timer.stop();
            dt_track += timer.elapsedSeconds();
            avg_track += ntracked_features;

            // Check inliers with essential
            cv::Mat cvMask;
            std::vector<cv::Point2f> p2f_last_track, p2f_inc_track;
            for (auto & match: last_map_inc){
                p2f_last_track.push_back(frame_last.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_track.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
            }
            computeEssential(K, p2f_last_track, p2f_inc_track, cvMask);
            inliers_track += 100 * cv::countNonZero(cvMask)/p2f_last_track.size();

            if (config.debug){
                std::cout << "Number of tracks" << std::endl;
                std::cout << ntracked_features << std::endl;

                // Image display for debug
                cv::Mat img_matches;
                for (auto & match: last_map_inc){
                    cv::DMatch dmatch(match.first,match.second,1);
                    good_matches.push_back(dmatch);
                }
                cv::drawMatches(img_last, frame_last.getCvKeyPointsVector(), img_inc, frame_inc.getCvKeyPointsVector(), 
                        good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }
        }

        if (config.enable_matcher){
            frame_inc.reset();
            frame_inc.setImg(img_inc);
            // detect features for Incoming
            parallelDetectAndCompute(frame_inc, detector, config.nrows, config.ncols);
            
            timer.start();
            std::map<int, int> last_map_inc;
            int nmatched_features = match(frame_last, frame_inc, last_map_inc,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
            timer.stop();
            dt_match += timer.elapsedSeconds();
            avg_match += nmatched_features;

            // Check inliers with essential
            cv::Mat cvMask;
            std::vector<cv::Point2f> p2f_last_match, p2f_inc_match;
            for (auto & match: last_map_inc){
                p2f_last_match.push_back(frame_last.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_match.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
            }
            computeEssential(K, p2f_last_match, p2f_inc_match, cvMask);
            inliers_match += 100 * cv::countNonZero(cvMask)/p2f_last_match.size();


            if (config.debug){
                std::cout << "Number of matches" << std::endl;
                std::cout << nmatched_features << std::endl;

                // Image Display
                cv::Mat img_matches;
                good_matches.clear();
                for (auto & match: last_map_inc){
                    cv::DMatch dmatch(match.first,match.second,1);
                    good_matches.push_back(dmatch);
                }
                cv::drawMatches(img_last, frame_last.getCvKeyPointsVector(), img_inc, frame_inc.getCvKeyPointsVector(), 
                        good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }
        }

        // Redetect to make frame to frame correspondence
        img_last = img_inc;
        frame_last.reset();
        frame_last.setImg(img_last);
        timer.start();
        parallelDetectAndCompute(frame_last, detector, config.ncols, config.nrows);
        timer.stop();
        dt_detect += timer.elapsedSeconds();
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

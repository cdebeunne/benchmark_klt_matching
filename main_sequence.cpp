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

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <filesystem>

std::map<int, int> merge_map(std::map<int, int> prev_map_curr, std::map<int, int> curr_map_next){
    std::map<int, int> prev_map_next;
    for (auto &match : prev_map_curr){
        if (curr_map_next.count(match.second) != 0){
            prev_map_next[match.first] = curr_map_next[match.second];
        }
    }
    return prev_map_next;
}


int main(int argc, char** argv){

    // initialize K
    Eigen::Matrix3f K;
    K << 458.654, 0, 367.215,
         0, 457.296, 248.375,
         0, 0, 1;

    // Load config
    Config config = readParameterFile(std::filesystem::current_path().string()+"/../"+"param.yaml");

    // path and loader of the EUROC sequence
    std::vector<std::string> img_list = EUROC_img_loader(config.dataset_path);
    cv::Mat img_inc, img_last, img_origin;

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
    Frame frame_origin, frame_last, frame_inc;
    std::map<int, int> origin_map_last, origin_map_inc, last_map_inc;

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

        // Origin init
        if (counter == 0){
            img_origin = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            frame_origin.setImg(img_origin);
            parallelDetectAndCompute(frame_origin, detector, config.nrows, config.ncols);
            frame_last = frame_origin;
            match(frame_origin, frame_last, origin_map_last,
                  config.matcher_width, config.matcher_height, config.threshold_matching);

            results << 0 << "," << origin_map_last.size() << ",\n";
            counter ++;
            continue;
        }

        img_inc = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        frame_inc.reset();
        frame_inc.setImg(img_inc);

        if (config.enable_tracker){
            last_map_inc.clear();
            int ntracked_features = track(frame_last, frame_inc, last_map_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
        

            if (ntracked_features < config.threshold_tracks){
                break;
                // parallelDetect(img_last, keypoints_last, detector, config.nrows, config.ncols);
                // ntracked_features = track(img_last, img_inc, keypoints_last, keypoints_inc,
                //                     config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                //                     config.klt_max_err);
                // keypoints_origin = keypoints_last;
                // origin_pairs_last = make_corespondence(keypoints_origin, keypoints_last);
            }

            // Filtering with Essential Matrix
            cv::Mat cvMask;
            origin_map_inc = merge_map(origin_map_last, last_map_inc);
            std::vector<cv::Point2f> p2f_origin_track, p2f_inc_track;
            for (auto & match: origin_map_inc){
                p2f_origin_track.push_back(frame_origin.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_track.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
            }
            if(!computeEssential(K, p2f_origin_track, p2f_inc_track, cvMask))break;
            
            // Removing the matches that are outliers 
            int k = 0;
            for (auto & match : origin_map_inc){
                if (cvMask.at<bool>(k) == 0){
                    origin_map_inc.erase(match.first);
                }
                k++;
            }

            // Incoming is now last
            origin_map_last = origin_map_inc;

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";

        }

        if (config.enable_matcher){
            // DetectionAndCompute & storage for repopulation
            parallelDetectAndCompute(frame_inc, detector, config.nrows, config.ncols);
            last_map_inc.clear();
            int nmatched_features = match(frame_last, frame_inc, last_map_inc,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
        

            if (nmatched_features < config.threshold_tracks){
                break;
            }

            // Filtering with Essential Matrix
            cv::Mat cvMask;
            origin_map_inc = merge_map(origin_map_last, last_map_inc);
            std::vector<cv::Point2f> p2f_origin_match, p2f_inc_match;
            for (auto & match: origin_map_inc){
                p2f_origin_match.push_back(frame_origin.getKeyPointIdx(match.first)._cvKeyPoint.pt);
                p2f_inc_match.push_back(frame_inc.getKeyPointIdx(match.second)._cvKeyPoint.pt);
            }
            if(!computeEssential(K, p2f_origin_match, p2f_inc_match, cvMask))break;
            
            // Removing the matches that are outliers
            int k = 0;
            for (auto & match : origin_map_inc){
                if (cvMask.at<bool>(k) == 0){
                    origin_map_inc.erase(match.first);
                }
                k++;
            }

            // Incoming is now Last
            origin_map_last = origin_map_inc;

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";
        }

        // Set inc as last (only for image this time)
        frame_last = frame_inc;
        counter++;
    }

    results.close();
}
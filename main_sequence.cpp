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

using feature_pair = std::pair<cv::KeyPoint,cv::KeyPoint>;

std::vector<feature_pair> make_corespondence(std::vector<cv::KeyPoint> kps_prev, std::vector<cv::KeyPoint> kps_curr){
    std::vector<feature_pair> features_corespondence;
    for (size_t i=0; i<kps_prev.size(); i++){
        features_corespondence.push_back(std::make_pair(kps_prev.at(i), kps_curr.at(i)));
    }
    return features_corespondence;
}

std::vector<feature_pair> edit_corespondences(std::vector<feature_pair> origin_pairs_prev, std::vector<feature_pair> prev_pairs_curr){
    std::vector<feature_pair> origin_pairs_curr;
    cv::KeyPoint kp_prev, kp_curr, kp_origin;
    feature_pair origin_p_curr;

    for (size_t i=0; i<prev_pairs_curr.size(); i++){
        kp_prev = prev_pairs_curr.at(i).first;
        kp_curr = prev_pairs_curr.at(i).second;

        for (size_t k=0; k<origin_pairs_prev.size(); k++){
            kp_origin = origin_pairs_prev.at(k).first;

            if (kp_prev.pt == origin_pairs_prev.at(k).second.pt){    
                origin_p_curr = std::make_pair(kp_origin, kp_curr);
                origin_pairs_curr.push_back(origin_p_curr);
                break;
            }
        }
    }
    return origin_pairs_curr;
}

void corespondences_to_keypoints(std::vector<feature_pair> prev_pairs_curr, 
                                 std::vector<cv::KeyPoint> &keypoints_prev, 
                                 std::vector<cv::KeyPoint> &keypoints_curr){
    keypoints_curr.clear();
    keypoints_prev.clear();
    for (auto & pair : prev_pairs_curr){
        keypoints_prev.push_back(pair.first);
        keypoints_curr.push_back(pair.second);
    }
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
    cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create(config.npoints,
                                            config.scale_factor,
                                            config.nlevels_pyramids,
                                            31, 0, 2, cv::ORB::FAST_SCORE, 31, 20);
    std::vector<cv::KeyPoint> keypoints_inc, keypoints_last, keypoints_origin;
    cv::Mat descriptors_inc, descriptors_last, descriptors_origin;
    std::vector<feature_pair> origin_pairs_last, origin_pairs_inc, last_pairs_inc;

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
            parallelDetect(img_origin, keypoints_origin, detector, config.nrows, config.ncols);
            descriptor->compute(img_origin, keypoints_origin, descriptors_origin);
            img_last = img_origin;
            descriptors_last = descriptors_origin;
            keypoints_last = keypoints_origin;
            origin_pairs_last = make_corespondence(keypoints_origin, keypoints_last);

            results << 0 << "," << keypoints_last.size() << ",\n";
            counter ++;
            continue;
        }

        img_inc = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        if (config.enable_tracker){
            int ntracked_features = track(img_last, img_inc, keypoints_last, keypoints_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
        

            if (ntracked_features < config.threshold_tracks){
                parallelDetect(img_last, keypoints_last, detector, config.nrows, config.ncols);
                ntracked_features = track(img_last, img_inc, keypoints_last, keypoints_inc,
                                    config.tracker_width, config.tracker_height, config.nlevels_pyramids_klt,
                                    config.klt_max_err);
                keypoints_origin = keypoints_last;
                origin_pairs_last = make_corespondence(keypoints_origin, keypoints_last);
            }

            // Edit pairs
            last_pairs_inc = make_corespondence(keypoints_last, keypoints_inc);
            origin_pairs_inc = edit_corespondences(origin_pairs_last, last_pairs_inc);

            // Filtering with Essential Matrix
            cv::Mat cvMask;
            corespondences_to_keypoints(origin_pairs_inc, keypoints_origin, keypoints_inc);
            if(!computeEssential(K, keypoints_origin, keypoints_inc, cvMask))break;

            if (config.debug){
                std::cout << "Number of tracks" << std::endl;
                std::cout << ntracked_features << std::endl;

                // Image Display
                cv::Mat img_matches;
                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < keypoints_inc.size(); i++){
                    if (cvMask.at<bool>(i) == 1){
                        cv::DMatch match(i,i,1);
                        good_matches.push_back(match);
                    }
                }
                cv::drawMatches(img_origin, keypoints_origin, img_inc, keypoints_inc, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }
            
            // Incoming is now Last
            keypoints_last.clear();
            for (size_t k=0; k<keypoints_inc.size(); k++){
                if (cvMask.at<bool>(k) == 1){
                    keypoints_last.push_back(keypoints_inc.at(k));
                }
            }
            origin_pairs_last = origin_pairs_inc;

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";

        }

        if (config.enable_matcher){
            // DetectionAndCompute & storage for repopulation
            parallelDetectAndCompute(img_inc, keypoints_inc, descriptors_inc, detector, config.nrows, config.ncols);
            std::vector<cv::KeyPoint> keypoints_inc_repop = keypoints_inc;
            cv::Mat descriptors_inc_repop = descriptors_inc;

            int nmatched_features = match(keypoints_last, keypoints_inc, descriptors_last, descriptors_inc,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
        

            if (nmatched_features < config.threshold_tracks){
                parallelDetectAndCompute(img_last, keypoints_last, descriptors_last, detector, config.nrows, config.ncols);
                nmatched_features = match(keypoints_last, keypoints_inc_repop, descriptors_last, descriptors_inc_repop,
                                        config.matcher_width, config.matcher_height, config.threshold_matching);
                keypoints_inc = keypoints_inc_repop;
                descriptors_inc = descriptors_inc_repop;
                keypoints_origin = keypoints_last;
                origin_pairs_last = make_corespondence(keypoints_origin, keypoints_last);
            }

            // Edit pairs
            last_pairs_inc = make_corespondence(keypoints_last, keypoints_inc);
            origin_pairs_inc = edit_corespondences(origin_pairs_last, last_pairs_inc);

            // Filtering with Essential Matrix 
            cv::Mat cvMask;
            corespondences_to_keypoints(origin_pairs_inc, keypoints_origin, keypoints_inc);
            if(!computeEssential(K, keypoints_origin, keypoints_inc, cvMask))break;

            if (config.debug){
                std::cout << "Number of matches" << std::endl;
                std::cout << nmatched_features << std::endl;

                // Image Display
                cv::Mat img_matches;
                std::vector<cv::DMatch> good_matches;
                for (size_t i = 0; i < keypoints_inc.size(); i++){
                    if (cvMask.at<bool>(i) == 1){
                        cv::DMatch match(i,i,1);
                        good_matches.push_back(match);
                    }
                }
                cv::drawMatches(img_origin, keypoints_origin, img_inc, keypoints_inc, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                cv::imshow( "Good Matches", img_matches);
                cv::waitKey(0);
            }

            // Incoming is now Last
            keypoints_last.clear();
            for (size_t k=0; k<keypoints_inc.size(); k++){
                if (cvMask.at<bool>(k) == 1){
                    keypoints_last.push_back(keypoints_inc.at(k));
                }
            }
            origin_pairs_last = origin_pairs_inc;
            // Compute descriptors of filtered point (suboptimal)
            descriptor->compute(img_inc, keypoints_last, descriptors_last);

            results << counter << ","
                    << cv::countNonZero(cvMask) << ", \n";
        }

        // Set inc as last (only for image this time)
        img_last = img_inc;
        counter++;
    }

    results.close();
}
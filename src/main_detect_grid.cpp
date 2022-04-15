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
#include "ImgLoader.hpp"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <filesystem>

/*
Goal of this file:
Compare the time taken by detection on a whole image vs detect one separate image tiles in a for loop.
**/



int main(int argc, char** argv)
{

    if (argc < 2){
        std::cout << "Missing param file argument!" << std::endl;
        return -1;
    }
    // Load config
    std::string param_file = argv[1];
    std::cout << "parameter file full path: " << std::filesystem::current_path().string()+"/../"+param_file << std::endl;
    // Load config
    Config config = readParameterFile(std::filesystem::current_path().string()+"/../"+param_file);
    std::cout << "Config read successfully" << std::endl;

    std::vector<std::string> img_list = EUROC_img_loader(config.dataset_path);

    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(config.threshold_fast);

    std::fstream results;
    std::string results_path = "result_detect_grid.csv";
    results.open(results_path, std::fstream::out);
    results << "dt_detect_global,dt_detect_grid,nb_global,nb_grid\n";

    Timer timer;
    double dt_detect_global = 0.0;
    double dt_detect_grid = 0.0;
    int counter = 0;
    for (const auto & img_name : img_list){
        std::string img_path = config.dataset_path + "/data/" + img_name;
        if (counter > config.max_nb_frames) break;

        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        // FIRST OPTION: detect on the whole image
        timer.start();
        std::vector<cv::KeyPoint> keypoints_global;
        detector->detect(img, keypoints_global);
        timer.stop();
        dt_detect_global = timer.elapsedSeconds();


        timer.start();
        int cwidth = img.cols  / config.nb_cells_h;
        int cheight = img.rows / config.nb_cells_v;
        // std::cout << "img.cols: " << img.cols << std::endl;
        // std::cout << "img.rows: " << img.rows << std::endl;
        // std::cout << "cwidth: " << cwidth << std::endl;
        // std::cout << "cheight: " << cheight << std::endl;

        std::vector<cv::KeyPoint> keypoints_grid;
        for (int i=0; i < config.nb_cells_h; i++){
            for (int j=0; j < config.nb_cells_v; j++){
                // compute ROI (no copy of data -> fast)
                cv::Point2f top_left = cv::Point2f(i*cwidth, j*cheight);
                cv::Point2f bottom_right = cv::Point2f((i+1)*cwidth, (j+1)*cheight) ;
                cv::Mat img_roi(img, cv::Rect(top_left, bottom_right));
                // Everything is coherent!
                // std::cout << "top_left:     " << top_left << std::endl;
                // std::cout << "bottom_right: " << bottom_right << std::endl;


                std::vector<cv::KeyPoint> keypoints_roi;
                detector->detect(img_roi, keypoints_roi);
                for (auto kp: keypoints_roi){
                    keypoints_grid.push_back(kp);
                }
            }
        }
        timer.stop();
        dt_detect_grid = timer.elapsedSeconds();

        results << dt_detect_global << "," 
                << dt_detect_grid << ","
                << keypoints_global.size() << ","
                << keypoints_grid.size() << std::endl;

        counter++;
    }

    results.close();

    return 0;
}

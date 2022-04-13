#ifndef IMGLOADER_H
#define IMGLOADER_H

#include <fstream>
#include <vector>

std::vector<std::string> EUROC_img_loader(std::string data_path){

    std::vector<std::string> img_list;

    std::ifstream cam(data_path+"/data.csv");
    std::string header;
    std::getline(cam,header);
    for(std::string line;getline(cam,line);) {
        std::istringstream stream(line);
        std::string stampStr, path;
        getline(stream, stampStr, ',');
        getline(stream, path, ',');
        path.erase(remove_if(path.begin(), path.end(), ::isspace), path.end());
        img_list.push_back(path);
    }
    if (img_list.size() == 0){
        std::cout << "\n!! No images found with path " << data_path << std::endl;
    }
    return img_list;
}

#endif
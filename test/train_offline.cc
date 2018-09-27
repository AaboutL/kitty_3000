#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "facemarkLBF.h"
#include "utilities.h"

using namespace std;

void read_file(const std::string& line, const std::string& prefix, cv::Mat& image, cv::Rect& rect, std::vector<cv::Point2f>& facial_points);

int main(int argc,char ** argv)
{
    ext::FacemarkLBF::Params params = ext::FacemarkLBF::Params();
    params.cascade_face = "../models/haarcascade_frontalface_alt.xml";
    params.n_landmarks = 64;
//    params.initShape_n = 2;
    params.stages_n=7;
    params.tree_n=6;
    params.tree_depth=6;
//    params.bagging_overlap = 0.4;
    params.model_filename = "../models/hfy_1_0505_1000_64_aug.xml";
    params.save_model = true;
    Ptr<ext::FacemarkTrain> facemark = ext::FacemarkLBF::create(params);

    std::string prefix = "";
    std::ifstream infile("../data/combined_fileinfo_list_0505.txt");
    if(!infile.is_open()) {
        std::cout << "file not open" << std::endl;
        return 0;
    }
    int num_file = 0;
    while(!infile.eof()) {
        std::string line;
        getline(infile, line);
        if(line.empty()) continue;
        cv::Mat image; cv::Rect rect;
        std::vector<cv::Point2f> facial_points;
        std::vector<cv::Point2f> shorter_points;
        read_file(line, prefix, image, rect, facial_points);
        std::vector<cv::Point2f>::iterator iter = facial_points.begin() + 19;
        shorter_points.assign(iter, facial_points.end());

        cv::Mat gray;
        if(image.empty()) continue;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

        facemark->addTrainingSample(gray, shorter_points, rect);
//        cv::Mat aug_img;
//        std::vector<cv::Point2f> aug_points;
//        cv::Rect aug_rect;
//        augment_transform(gray, shorter_points, aug_img, aug_points, aug_rect);
//        facemark->addTrainingSample(aug_img, aug_points, rect);
//        std::cout << "aug" << std::endl;
        num_file++;
//        if(num_file++ == 20)
//            break;
    }
    std::cout << "feed data end, " << num_file << "fed!" << std::endl;

	cout<<"training..."<<endl;
	facemark->training();

	return EXIT_SUCCESS;
}

void read_file(const std::string& line, const std::string& prefix, cv::Mat& image, cv::Rect& rect, std::vector<cv::Point2f>& facial_points) {
    size_t pos = line.find(' ');
    std::string name = std::string(line, 0, pos);
    std::string img_name = prefix + name;
    image = cv::imread(img_name);
    std::cout << img_name << std::endl;

    // find first ')'
    size_t pos_bbox = line.find(')');
    std::string bbox = std::string(line, pos + 2, pos_bbox - pos - 2);
    std::istringstream ss1(bbox);
    std::string x, y, width, height;
    ss1 >> x >> y >> width >> height;
    rect = cv::Rect(atoi(x.c_str()), atoi(y.c_str()), atoi(width.c_str()), atoi(height.c_str()));

    //find second ')'
    size_t pos_lm = line.find(')', pos_bbox + 3);
    std::string landmark = std::string(line, pos_bbox + 3, pos_lm - pos_bbox -3);
    std::istringstream ss2(landmark);
    while(!ss2.eof()){
        std::string x1, y1;
        ss2 >> x1 >> y1;
        facial_points.push_back(cv::Point2f((float)atof(x1.c_str()), (float)atof(y1.c_str())));
    }
    std::cout << "read file end" << std::endl;
}

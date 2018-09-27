#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
//#include "FaceDetection.h"
//#include "facemarkLBF.h"

using namespace std;

bool detector(cv::InputArray img,cv::OutputArray faces,void *);
void read_file(const std::string& line, const std::string& prefix, cv::Mat& image, cv::Rect& rect, std::vector<cv::Point2f>& facial_points);

int main(int argc,char ** argv)
{
    cv::face::FacemarkLBF::Params params = cv::face::FacemarkLBF::Params();
    params.cascade_face = "../models/haarcascade_frontalface_alt.xml";
    params.model_filename = "../models/ibugs_lbf_cv.xml";
    params.save_model = true;
    cv::Ptr<cv::face::FacemarkTrain> facemark = cv::face::FacemarkLBF::create(params);
//    facemark->setFaceDetector(detector);
	cv::String imageFiles = "../data/ibugs_images.txt";
	cv::String ptsFiles = "../data/ibugs_points.txt";
//    cv::String imageFiles = "/home/slam/workspace/DL/align_3000fps_cv/data/300W-LP_images_helen.txt";
//    cv::String ptsFiles = "/home/slam/workspace/DL/align_3000fps_cv/data/300W-LP_points_helen.txt";
	std::vector<cv::String> images_train;
	std::vector<cv::String> landmarks_train;
	cv::face::loadDatasetList(imageFiles, ptsFiles, images_train, landmarks_train);
    std::cout << images_train.size() << std::endl;
	cv::Mat image;
	std::vector<cv::Point2f> facial_points;
	for(size_t i = 0; i < images_train.size(); i++) {
        std::cout << images_train[i] << std::endl;
		image = cv::imread(images_train[i].c_str());
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        cv::resize(gray, gray, gray.size()/2);
        cv::face::loadFacePoints(landmarks_train[i], facial_points); //pts
        for (auto &pt:facial_points) pt = pt/2;
        std::cout << "add sample: " << i << std::endl;
        facemark->addTrainingSample(gray, facial_points);
//        cv::imshow("gray", gray);
//        cv::waitKey(0);
	}

	cout<<"training..."<<endl;
	facemark->training();

	return EXIT_SUCCESS;
}

bool detector(cv::InputArray img,cv::OutputArray faces,void *)
{
    std::vector<cv::Mat> channels(3,img.getMat());
    cv::Mat image;
    cv::merge(channels,image);
    faces.clear();
//    static FaceDetection fd;
//    Mat(fd.detect(image)).copyTo(faces);
    return true;
}

void read_file(const std::string& line, const std::string& prefix, cv::Mat& image, cv::Rect& rect, std::vector<cv::Point2f>& facial_points) {
    size_t pos = line.find(' ');
    std::string name = std::string(line, 0, pos);
    std::string img_name = prefix + name;
    image = cv::imread(img_name);
//    cv::imshow("test", image);
//    cv::waitKey(30);

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
        std::string x, y;
        ss2 >> x >> y;
        std::cout << x << "  " << y << std::endl;
        facial_points.push_back(cv::Point2f((float)atof(x.c_str()), (float)atof(y.c_str())));
    }
}

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
#include "facemarkLBF.h"

using namespace std;

bool detector(InputArray img,OutputArray faces,void *);
void read_file(const std::string& line, const std::string& prefix, cv::Mat& image, cv::Rect& rect, std::vector<cv::Point2f>& facial_points);

int main(int argc,char ** argv)
{
    Ptr<ext::FacemarkTrain> facemark = ext::FacemarkLBF::create();
    facemark->loadModel("../models/train_lbf_0422(all-2).xml");

//    std::string prefix = "/home/slam/dataset/DL/alignment/300W/01_Indoor/";
//    std::ifstream infile("/home/slam/workspace/DL/align_3000fps_cv/data/test.txt");
    std::string prefix = "";
    std::ifstream infile("/home/slam/nfs72/face/alignment0422/facepp_result.txt");
    if(!infile.is_open()) {
        std::cout << "file not open" << std::endl;
        return 0;
    }
    int file_num = 0;
    while(!infile.eof()) {

        std::string line;
        getline(infile, line);
        if(file_num++ < 10000)
            continue;
        if(line.empty()) continue;
        cv::Mat image;
        cv::Rect rect;
        std::vector<cv::Point2f> facial_points;
        read_file(line, prefix, image, rect, facial_points);
        cv::Mat gray;
//        cv::imshow("image", image);
//        cv::waitKey(0);
        if(image.empty()) continue;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

        std::vector<cv::Rect> faces;
        std::vector<std::vector<cv::Point2f> > landmarks, groundtruth;
        rect.width += 20;
        rect.height += 20;
        faces.push_back(rect);

        bool sucess = facemark->fit(gray, faces, landmarks);
        groundtruth.push_back(facial_points);
        if(sucess) {
            for(int i = 0; i < landmarks[0].size(); i++) {
                cv::circle(image, landmarks[0][i], 2, cv::Scalar(0, 0, 255));
                cv::circle(image, groundtruth[0][i], 2, cv::Scalar(255, 0, 0));
                cv::rectangle(image, faces[0], cv::Scalar(255, 0, 0));
            }
        }
        cv::imshow("face", image);
        if(waitKey(0) == 27) break;

//        if(file_num++ == 10)
//            break;
    }

	return EXIT_SUCCESS;
}

bool detector(InputArray img,OutputArray faces,void *)
{
    std::vector<Mat> channels(3,img.getMat());
    Mat image;
    merge(channels,image);
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
    std::cout << img_name << std::endl;
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
//        std::cout << x << "  " << y << std::endl;
        facial_points.push_back(cv::Point2f((float)atof(x.c_str()), (float)atof(y.c_str())));
    }
    std::cout << "read file end" << std::endl;
}

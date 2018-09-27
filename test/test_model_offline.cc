#include "facemarkLBF.h"
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    cv::CascadeClassifier faceDetector("/home/slam/workspace/DL/align_3000fps_cv/models/haarcascade_frontalface_alt.xml");
    cv::Ptr<ext::Facemark>  facemark = ext::FacemarkLBF::create();
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/ibugs_lbf.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/lbfmodel.yaml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/train_lbf_0422(all-1).xml");
    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_0505_1_64_no_aug.xml");
    cv::Mat frame, gray;
    std::ifstream infile("/home/slam/nfs72/face/facial_points/test_data/left_right.txt");
    if(!infile.is_open()) {
        std::cout << "file not open!" << std::endl;
        return 0;
    }
    while(!infile.eof()) {
        std::string line;
        getline(infile, line);
        if(line.empty()) continue;
        size_t pos = line.find(' ');
        size_t pos_l = line.find('(');
        size_t pos_r = line.find(')');
        std::string file_path = std::string(line, 0, pos);
        std::cout << file_path << std::endl;
        std::string rect_str = std::string(line, pos_l+1, pos_r-pos_l-1);
        std::stringstream ss(rect_str);
        int x, y, w, h;
        ss >> x; ss >> y; ss >> w; ss >> h;
        cv::Rect face = cv::Rect(x, y, w, h);
        frame = cv::imread(file_path);
        std::vector<cv::Rect> faces;
        faces.emplace_back(face);
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
//        faceDetector.detectMultiScale(gray, faces);
        std::vector<std::vector<cv::Point2f> > landmarks;
        bool sucess = facemark->fit(frame, faces, landmarks);
        std::cout << "sucess" << std::endl;
        if(sucess) {
            for(int i = 0; i < landmarks[0].size(); i++) {
                cv::circle(frame, landmarks[0][i], 2, cv::Scalar(0, 0, 255));
            }
        }
        std::string save_path;
        save_path = file_path.replace(46, 10, "landmark_hfy_0505_1_64_no_aug/left_right_lm");
        std::cout << save_path<< std::endl;
//        cv::imshow("face", frame);
        cv::imwrite(save_path, frame);
//        if(waitKey(0) == 27) break;
    }
    return 0;
}
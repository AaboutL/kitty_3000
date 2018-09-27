#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

int main(int argc, char** argv) {
    cv::CascadeClassifier faceDetector("/home/slam/workspace/DL/align_3000fps_cv/models/haarcascade_frontalface_alt.xml");
    cv::Ptr<cv::face::Facemark>  facemark = cv::face::FacemarkLBF::create();
//    facemark->loadModel("../models/lbfmodel.yaml");
//    facemark->loadModel("../models/ibugs_lbf.xml");
    facemark->loadModel("../models/train_lbf_0422(1000).xml");
    cv::Mat frame, gray;
    std::ifstream infile("/home/slam/nfs72/face/alignment0422/facepp_result_test.txt");
    if(!infile.is_open()) {
        std::cout << "file not open!" << std::endl;
        return 0;
    }
    while(!infile.eof()) {
        std::string line;
        getline(infile, line);
        if(line.empty()) continue;
        size_t pos = line.find(' ');
        std::string file_path = std::string(line, 0, pos);
        frame = cv::imread(file_path);
        std::vector<cv::Rect> faces;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        faceDetector.detectMultiScale(gray, faces);
        std::vector<std::vector<cv::Point2f> > landmarks;
        bool sucess = facemark->fit(frame, faces, landmarks);
        std::cout << "sucess" << std::endl;
        if(sucess) {
            for(int i = 0; i < landmarks[0].size(); i++) {
                cv::circle(frame, landmarks[0][i], 2, cv::Scalar(0, 0, 255));
            }
        }
        cv::imshow("face", frame);
        if(cv::waitKey(0) == 27) break;
    }
    return 0;
}
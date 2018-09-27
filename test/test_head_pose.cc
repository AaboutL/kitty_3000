#include "facemarkLBF.h"
#include <fstream>
#include <ctime>
#include "head_pose_estimator.h"
#include "utilities.h"

#define TIMER_BEGIN { double __time__ = (double)getTickCount();
#define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
#define TIMER_END   }

int main(int argc, char** argv) {
    cv::CascadeClassifier faceDetector("../models/haarcascade_frontalface_alt.xml");
    ext::FacemarkLBF::Params params = ext::FacemarkLBF::Params();
    cv::Ptr<ext::Facemark>  facemark = ext::FacemarkLBF::create();
    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_zb_0503.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_2_zb_0503_rotate.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_2_zb_0504_64.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_0504_1_64.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_0505_1_64_no_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_0505_1000_64_no_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_0505_1000_64_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/train_lbf_0422(all-1).xml");
//    cv::VideoCapture cam(0);
    cv::VideoCapture cam("/home/slam/nfs72/face/ljj.avi");
    cv::Mat frame, gray;
    bool flag = cam.read(frame);
    while(cam.read(frame)) {
        // face++ headpose": {"yaw_angle": -4.489544, "pitch_angle": -5.9814334, "roll_angle": -3.8065782}
        frame = cv::imread("/home/slam/nfs72/face/facial_points/alignment/xiner_20180424_zhuzhixiang/1/data/cam0/data/1524539678200866601.jpg");
        cv::Rect rect = cv::Rect(228,201,196,196);
        HeadPoseEstimator estimator(frame);
        std::vector<cv::Rect> faces;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
//        faceDetector.detectMultiScale(gray, faces);
        faces.emplace_back(rect);
        std::vector<std::vector<cv::Point2f> > landmarks;
        bool sucess = false;
        TIMER_BEGIN
            sucess = facemark->fit(frame, faces, landmarks);
            std::cout << "Current frame cost: " << TIMER_NOW << std::endl;
        TIMER_END
        landmark_face_bboxes_ratio(faces[0], landmarks[0], sucess, 0.7);
        std::vector<cv::Point2d> five_points;
        five_points.emplace_back(landmarks[0][44]);
        five_points.emplace_back(landmarks[0][45]);
        five_points.emplace_back(landmarks[0][59]);
        five_points.emplace_back(landmarks[0][65]);
        five_points.emplace_back(landmarks[0][71]);
        cv::Vec3d euler = estimator.estimate_pose(five_points);
        std::cout << "current angle: " << euler[0] << ' ' << euler[1] << ' ' << euler[2] << std::endl;
        if(sucess) {
            for(int i = 0; i < landmarks[0].size(); i++) {
                cv::circle(frame, landmarks[0][i], 2, cv::Scalar(0, 0, 255));
            }
        }
        cv::imshow("face", frame);
        if(waitKey(1) == 27) break;
    }
    return 0;
}
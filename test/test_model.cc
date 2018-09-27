#include "facemarkLBF.h"
#include <fstream>
#include <ctime>

#define TIMER_BEGIN { double __time__ = (double)getTickCount();
#define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
#define TIMER_END   }

int main(int argc, char** argv) {
    cv::CascadeClassifier faceDetector("../models/haarcascade_frontalface_alt.xml");
    ext::FacemarkLBF::Params params = ext::FacemarkLBF::Params();
//    params.cascade_face = "../models/haarcascade_frontalface_alt.xml";
//    params.n_landmarks = 64;
//    params.initShape_n = 2;
//    params.stages_n=7;
//    params.tree_n=7;
//    params.tree_depth=6;
//    params.bagging_overlap = 0.4;
//    params.model_filename = "../models/hfy_2_zb_0504_64.xml";
//    cv::Ptr<ext::Facemark>  facemark = ext::FacemarkLBF::create(params);
    cv::Ptr<ext::Facemark>  facemark = ext::FacemarkLBF::create();
    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_zb_0503.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_2_zb_0503_rotate.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_2_zb_0504_64.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_0504_1_64.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_0505_1_64_no_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_0505_1000_64_no_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/hfy_1_0505_1000_64_aug.xml");
//    facemark->loadModel("/home/slam/workspace/DL/align_3000fps_cv/models/train_lbf_0422(all-1).xml");
    cv::VideoCapture cam(0);
//    cv::VideoCapture cam("/home/slam/nfs72/face/ljj.avi");
    cv::Mat frame, gray;
    while(cam.read(frame)) {
        std::vector<cv::Rect> faces;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        faceDetector.detectMultiScale(gray, faces);
        std::vector<std::vector<cv::Point2f> > landmarks;
        bool sucess = false;
        TIMER_BEGIN
            sucess = facemark->fit(frame, faces, landmarks);
            std::cout << "Current frame cost: " << TIMER_NOW << std::endl;
        TIMER_END
        if(!faces.empty() && !landmarks.empty()) {
            cv::Rect bbox_face = faces[0];
            cv::Rect bbox_points = cv::boundingRect(landmarks[0]);
            float face_area = bbox_face.area();
            float points_area = bbox_points.area();
            float ratio = points_area / face_area;
            if (ratio > 0.7) {
                sucess = true;
            }
        }
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
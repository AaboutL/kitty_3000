//
// Created by slam on 18-5-3.
//

#ifndef ALIGN_3000FPS_CV_UTILITIES_H
#define ALIGN_3000FPS_CV_UTILITIES_H

#include <opencv2/opencv.hpp>
#include <ctime>
#include <vector>

using namespace cv;

Point2f point_transform(Point2f pt, Point2f center, float s, float rot)
{
    float rad = rot * M_PI / 180;
    Mat affine;
    //translate image to make center at the origin
    Mat translation1 = Mat::eye(cv::Size(3,3),CV_32FC1);
    translation1.at<float>(0,2) = -center.x;
    translation1.at<float>(1,2) = -center.y;
    //counter-clockwise rotate image for rot angle
    Mat rotate = Mat::eye(cv::Size(3,3),CV_32FC1);
    rotate.at<float>(0,0) = cos(rad);	rotate.at<float>(0,1) = -sin(rad);
    rotate.at<float>(1,0) = sin(rad);	rotate.at<float>(1,1) = cos(rad);
    //scale image
    affine = s * rotate * translation1;
    //translate image to let the new upper left at the origin
    Mat translation2 = Mat::eye(cv::Size(3,3),CV_32FC1);
    translation2.at<float>(0,2) = center.x;
    translation2.at<float>(1,2) = center.y;
    affine = translation2 * affine;
    //transform coordinate
    affine.at<float>(2,2) = 1;
    Mat pt_ = Mat::ones(cv::Size(1,3),CV_32FC1);
    pt_.at<float>(0,0) = pt.x;
    pt_.at<float>(1,0) = pt.y;
    Mat npt_ = affine * pt_;
    return Point2f(npt_.at<float>(0,0),npt_.at<float>(1,0));
}

void augment_transform(const cv::Mat& ori_img, const std::vector<cv::Point2f>& facial_points,
               cv::Mat& ret_img, std::vector<cv::Point2f>& ret_points, cv::Rect& ret_rect) {
    cv::RNG rng((unsigned)time(NULL));
    double angle = rng.uniform(-30, 30);
    int width = ori_img.cols;
    int height = ori_img.rows;
    cv::Point2f center = cv::Point2f(width/2, height/2);
    cv::Mat trans = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(ori_img, ret_img, trans, ori_img.size());

    // rotate points
    /*
    for(const auto &pt : facial_points) {
//        cv::Point2f tmp = rot * pt;
        std::cout << "pt: " << pt.x << ' ' << pt.y << std::endl;
//        float x = trans.at<float>(0,0)*pt.x + trans.at<float>(0, 1)*pt.y + trans.at<float>(0,2);
//        float y = trans.at<float>(1,0)*pt.x + trans.at<float>(1,1)*pt.y + trans.at<float>(1,2);
        int alpha = cos(rad);
        int beta = sin(rad);
        float x = alpha * pt.x + beta * pt.y + (1 - alpha) * center.x - beta * center.y;
        float y = -beta * pt.x + alpha * pt.y + beta * center.x + (1 - alpha) * center.y;
        cv::Point2f tmp = cv::Point2f(x, y);
        std::cout << x << "  " << y << std::endl;
        ret_points.push_back(tmp);
        cv::circle(ret_img, tmp, 2, (255, 0, 0));
    }
    */
    for (auto &pt : facial_points) {
        cv::Point2f tmp = point_transform(pt, center, 1.0, -angle);
        ret_points.push_back(tmp);
//        cv::circle(ret_img, tmp, 2, (255, 0, 0));
    }

    ret_rect = cv::boundingRect(ret_points);
//    cv::rectangle(ret_img, cv::Point2f(ret_rect.x, ret_rect.y), cv::Point2f(ret_rect.x + ret_rect.width, ret_rect.y + ret_rect.height),(0, 255, 0));

//    cv::imshow("ret_img", ret_img);
//    cv::waitKey(0);
}

void landmark_face_bboxes_ratio(const cv::Rect& face, const std::vector<cv::Point2f>& landmark, bool& is_success, const float threshold) {
    if(face.empty() || landmark.empty() || is_success==false)
        is_success = false;
    float face_area = face.area();
    cv::Rect points_bbox = cv::boundingRect(landmark);
    float points_area = points_bbox.area();
    float ratio = points_area / face_area;
    if(ratio > threshold)
        is_success = true;
    else
        is_success = false;
}

void add_noise(const cv::Mat& ori_img, cv::Mat& out_img, int factor, double sigma){
    cv::RNG rng((unsigned)time(NULL));
    int MaxPixel = 255, MinPixel = 0;
    out_img = ori_img.clone();
    for(int i = 0; i < out_img.rows; i++) {
        for(int j = 0; j < out_img.cols; j++) {
            int rand_i = rng.uniform(-factor, factor);
            double rand_d = rng.gaussian(sigma);
            double tmp = out_img.at<uchar>(i, j) + rand_i*rand_d;
            if(tmp > MaxPixel) tmp = MaxPixel;
            if(tmp < MinPixel) tmp = MinPixel;
            out_img.at<uchar>(i, j) = tmp;
        }
    }
}

#endif //ALIGN_3000FPS_CV_UTILITIES_H

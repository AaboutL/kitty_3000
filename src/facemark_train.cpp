//
// Created by slam on 18-4-20.
//

#include "precomp.hpp"
#include "facemark_train.h"

/*dataset parser*/
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>     /* atoi */

namespace ext {
//    namespace face {

    using namespace std;

    CParams::CParams(cv::String s, double sf, int minN, cv::Size minSz, cv::Size maxSz) {
        cascade = s;
        scaleFactor = sf;
        minNeighbors = minN;
        minSize = minSz;
        maxSize = maxSz;

        if (!face_cascade.load(cascade)) {
            CV_Error_(cv::Error::StsBadArg, ("Error loading face_cascade: %s", cascade.c_str()));
        }
    }

    bool getFaces(cv::InputArray image, cv::OutputArray faces, CParams *params) {
        CV_Assert(params);
        cv::Mat gray;
        std::vector<cv::Rect> roi;

        cv::cvtColor(image.getMat(), gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        params->face_cascade.detectMultiScale(gray, roi, params->scaleFactor, params->minNeighbors, cv::CASCADE_SCALE_IMAGE,
                                              params->minSize, params->maxSize);

        cv::Mat(roi).copyTo(faces);
        return true;
    }

    bool loadDatasetList(cv::String imageList, cv::String groundTruth, std::vector<cv::String> &images, std::vector<cv::String> &landmarks) {
        std::string line;

        /*clear the output containers*/
        images.clear();
        landmarks.clear();

        /*open the files*/
        std::ifstream infile;
        infile.open(imageList.c_str(), std::ios::in);
        std::ifstream ss_gt;
        ss_gt.open(groundTruth.c_str(), std::ios::in);
        if ((!infile) || !(ss_gt)) {
            printf("No valid input file was given, please check the given filename.\n");
            return false;
        }

        /*load the images path*/
        while (getline(infile, line)) {
            images.push_back(line);
        }

        /*load the points*/
        while (getline(ss_gt, line)) {
            landmarks.push_back(line);
        }

        return true;
    }

    bool loadTrainingData(cv::String filename, std::vector<cv::String> &images, cv::OutputArray _facePoints, char delim, float offset) {
        std::string line;
        std::string item;
        std::vector<cv::Point2f> pts;
        std::vector<float> raw;

        // FIXIT
        std::vector<std::vector<cv::Point2f> > &facePoints =
                *(std::vector<std::vector<cv::Point2f> > *) _facePoints.getObj();

        std::ifstream infile;
        infile.open(filename.c_str(), std::ios::in);
        if (!infile) {
            CV_Error_(cv::Error::StsBadArg,
                      ("No valid input file was given, please check the given filename: %s", filename.c_str()));
        }

        /*clear the output containers*/
        images.clear();
        facePoints.clear();

        /*the main loading process*/
        while (getline(infile, line)) {
            std::istringstream ss(line); // string stream for the current line

            /*pop the image path*/
            getline(ss, item, delim);
            images.push_back(item);

            /*load all numbers*/
            raw.clear();
            while (getline(ss, item, delim)) {
                raw.push_back((float) atof(item.c_str()));
            }

            /*convert to opencv points*/
            pts.clear();
            for (unsigned i = 0; i < raw.size(); i += 2) {
                pts.push_back(cv::Point2f(raw[i] + offset, raw[i + 1] + offset));
            }
            facePoints.push_back(pts);
        } // main loading process

        return true;
    }

    bool loadTrainingData(cv::String imageList, cv::String groundTruth, std::vector<cv::String> &images, cv::OutputArray _facePoints,
                          float offset) {
        std::string line;
        std::vector<cv::Point2f> facePts;

        // FIXIT
        std::vector<std::vector<cv::Point2f> > &facePoints =
                *(std::vector<std::vector<cv::Point2f> > *) _facePoints.getObj();

        /*clear the output containers*/
        images.clear();
        facePoints.clear();

        /*load the images path*/
        std::ifstream infile;
        infile.open(imageList.c_str(), std::ios::in);
        if (!infile) {
            CV_Error_(cv::Error::StsBadArg,
                      ("No valid input file was given, please check the given filename: %s", imageList.c_str()));
        }

        while (getline(infile, line)) {
            images.push_back(line);
        }

        /*load the points*/
        std::ifstream ss_gt(groundTruth.c_str());
        while (getline(ss_gt, line)) {
            facePts.clear();
            loadFacePoints(line, facePts, offset);
            facePoints.push_back(facePts);
        }

        return true;
    }

    bool loadFacePoints(cv::String filename, cv::OutputArray points, float offset) {
        vector<cv::Point2f> pts;

        std::string line, item;
        std::ifstream infile(filename.c_str());

        /*pop the version*/
        std::getline(infile, line);
        assert(line.compare(0, 7, "version") == 0);

        /*pop the number of points*/
        std::getline(infile, line);
        assert(line.compare(0, 8, "n_points") == 0);

        /*get the number of points*/
        std::string item_npts;
        int npts;

        std::istringstream linestream(line);
        linestream >> item_npts >> npts;

        /*pop out '{' character*/
        std::getline(infile, line);

        /*main process*/
        int cnt = 0;
        std::string x, y;
        pts.clear();
        while (std::getline(infile, line) && cnt < npts) {
            cnt += 1;

            std::istringstream ss(line);
            ss >> x >> y;
            pts.push_back(cv::Point2f((float) atof(x.c_str()) + offset, (float) atof(y.c_str()) + offset));

        }
        cv::Mat(pts).copyTo(points);
        return true;
    }

    bool loadFacePoints(cv::String filename, cv::OutputArray points, int num_pts, char split_char, float offset) {
        vector<cv::Point2f> pts;

        std::string line, item;
        std::ifstream infile(filename.c_str());

        /*get the number of points*/
        int npts = num_pts;

        /*main process*/
        int cnt = 0;
        std::string x, y;
        pts.clear();
        while (std::getline(infile, line) && cnt < npts) {
            cnt += 1;
            if(split_char != NULL) {
                size_t pos = line.find(split_char);
                x = line.substr(0, pos);
                y = line.substr(pos + 1, pos);
            }
            else {
                std::istringstream ss(line);
                ss >> x >> y;
            }
            pts.push_back(cv::Point2f((float) atof(x.c_str()) + offset, (float) atof(y.c_str()) + offset));

        }

        cv::Mat(pts).copyTo(points);
        return true;
    }

    bool getFacesHAAR(cv::InputArray image, cv::OutputArray faces, const cv::String &face_cascade_name) {
        cv::Mat gray;
        std::vector<cv::Rect> roi;
        cv::CascadeClassifier face_cascade;
        assert(face_cascade.load(face_cascade_name) && "Can't loading face_cascade");
        cv::cvtColor(image.getMat(), gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        face_cascade.detectMultiScale(gray, roi, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        cv::Mat(roi).copyTo(faces);
        return true;
    }

    bool loadTrainingData(vector<cv::String> filename, vector<vector<cv::Point2f> >
    &trainlandmarks, vector<cv::String> &trainimages) {
        std::string img;
        std::vector<cv::Point2f> temp;
        std::string s;
        std::string tok;
        std::vector<std::string> coordinates;
        std::ifstream f1;
        for (unsigned long j = 0; j < filename.size(); j++) {
            f1.open(filename[j].c_str(), ios::in);
            if (!f1.is_open()) {
                cout << filename[j] << endl;
                CV_ErrorNoReturn(cv::Error::StsError, "File can't be opened for reading!");
                return false;
            }
            //get the path of the image whose landmarks have to be detected
            getline(f1, img);
            //push the image paths in the vector
            trainimages.push_back(img);
            img.clear();
            while (getline(f1, s)) {
                cv::Point2f pt;
                std::stringstream ss(s); // Turn the string into a stream.
                while (getline(ss, tok, ',')) {
                    coordinates.push_back(tok);
                    tok.clear();
                }
                pt.x = (float) atof(coordinates[0].c_str());
                pt.y = (float) atof(coordinates[1].c_str());
                coordinates.clear();
                temp.push_back(pt);
            }
            trainlandmarks.push_back(temp);
            temp.clear();
            f1.close();
        }
        return true;
    }

    void drawFacemarks(cv::InputOutputArray image, cv::InputArray points, cv::Scalar color) {
        cv::Mat img = image.getMat();
        std::vector<cv::Point2f> pts = points.getMat();
        for (size_t i = 0; i < pts.size(); i++) {
            cv::circle(img, pts[i], 3, color, -1);
        }
    }
//    } /* namespace face */
} /* namespace ext */

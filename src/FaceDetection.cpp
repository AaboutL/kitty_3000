#include "FaceDetection.h"
#include "../../dlib/dlib/pixel.h"
#include "../../dlib/dlib/opencv/cv_image.h"
#include "../../dlib/dlib/image_transforms/assign_image.h"
#include "../../dlib/dlib/image_transforms/interpolation.h"

FaceDetection::FaceDetection()
{
	deserialize("/home/slam/workspace/DL/align_3000fps_cv/models/mmod_human_face_detector.dat") >> net;
}

FaceDetection::~FaceDetection()
{
}

std::vector<Rect> FaceDetection::detect(Mat img)
{
	matrix<rgb_pixel> rgb;
	assign_image(rgb,cv_image<bgr_pixel>(img));
	float scale = 1;
	while(rgb.size() < 1800 * 1800) {
		pyramid_up(rgb);
		scale *= 2;
	}
	auto dets = net(rgb);
	std::vector<Rect> retVal;
	for(auto && objects : dets) {
		retVal.push_back(Rect(
			Point(objects.rect.left() / scale,objects.rect.top() / scale),
			Point(objects.rect.right() / scale,objects.rect.bottom() / scale)
		));
	}
	return retVal;
}

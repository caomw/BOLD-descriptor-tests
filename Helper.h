#ifndef HELPER_H
#define HELPER_H

#include "utils.h"
#include "bold.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class Helper
{
public:
    Helper(void);
    ~Helper();
    vector<Mat> ComputePatches(vector<KeyPoint>, Mat img);
    void ComputeBinaryDescriptors(vector<Mat> patches, Mat &descriptors, Mat &masks);

private:
    Mat GetPatch(Mat img, KeyPoint keypoint);
    CvSize patchSize;
};

#endif // HELPER_H

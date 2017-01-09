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
    void computePatches(vector<KeyPoint> &, Mat &img, vector<Mat> &patches);
    void computeBinaryDescriptors(vector<Mat>& patches, Mat& descriptors, Mat& masks);
    void getMatches();

    ///uses knn brute force matcher
    void findMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches, NormTypes norm, const float ratio);

    ///uses regular brute force matcher
    void findMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches);

    void saveKeypointsToFile(string filename, vector<KeyPoint> &keypoints);
    void drawFoundMatches(Mat& img1, vector<KeyPoint>& keypoints1, Mat& img2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, Mat& imgMatches);

private:
    Point getMinimumHammingDistancePoint(vector<DMatch> &vec);
    Mat getPatch(Mat &img, KeyPoint &keypoint, const CvSize &patchSize);
    int hampopmasked(uchar *a,uchar *ma,uchar *b,uchar *mb);
    int hampop(uchar *a,uchar *b);
};

#endif // HELPER_H

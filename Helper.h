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
    void ComputePatches(vector<KeyPoint> &, Mat &img, vector<Mat> &patches);
    void ComputeBinaryDescriptors(vector<Mat>& patches, Mat& descriptors, Mat& masks);
    void GetMatches();

    ///uses knn brute force matcher
    void FindMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches, NormTypes norm, const float ratio);

    ///uses regular brute force matcher
    void FindMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches);

    void SaveKeypointsToFile(string filename, vector<KeyPoint> &keypoints);
    void DrawMatches(Mat& img1, vector<KeyPoint>& keypoints1, Mat& img2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, Mat& imgMatches);
private:
    Point GetMinimumHammingDistancePoint(vector<DMatch> &vec);
    Mat GetPatch(Mat &img, KeyPoint &keypoint, const CvSize &patchSize);
    int Hampopmasked(uchar *a,uchar *ma,uchar *b,uchar *mb);
    int Hampop(uchar *a,uchar *b);
};

#endif // HELPER_H

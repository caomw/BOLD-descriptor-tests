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
    vector<myMatch> ComputePatches(vector<KeyPoint>, Mat img);
    void ComputeBinaryDescriptors(vector<myMatch>& patches, string filename, int descNum);
    void GetMatches();
    void FindMatches(vector<myMatch> desca, vector<myMatch> descb, vector<vector<Point> > &finalMatches);
    void SaveKeypointsToFile(string filename, vector<KeyPoint> keypoints);
private:
    Mat GetPatch(Mat img, KeyPoint keypoint);
    CvSize patchSize;
    int Hampopmasked(uchar *a,uchar *ma,uchar *b,uchar *mb);
    int Hampop(uchar *a,uchar *b);
};

#endif // HELPER_H

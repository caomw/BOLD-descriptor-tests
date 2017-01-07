#include "utils.h"
#include "bold.hpp"
#include "Helper.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    //-- Read images and resize them
    Mat img1 = imread("s1.jpg", IMREAD_GRAYSCALE );
    Mat img2 = imread("s22.jpg", IMREAD_GRAYSCALE );
    resize(img1, img1, Size(600, 800));
    resize(img2, img2, Size(600, 800));

    //-- Detect keypoints using SURF Detector
    Ptr<Feature2D> detectHandler = SURF::create();
    vector<KeyPoint> keypoints1, keypoints2;
    detectHandler->detect(img1, keypoints1);
    detectHandler->detect(img2, keypoints2);

    //-- Init Helper
    Helper ImageHelper;

    //-- Get patches of compared images
    vector<Mat> patches1, patches2;
    ImageHelper.ComputePatches(keypoints1, img1, patches1);
    ImageHelper.ComputePatches(keypoints2, img2, patches2);

    //-- Describe keypoints
    Mat descriptors1, descriptors2;
    Mat masks1, masks2;
    //Ptr<Feature2D> describeHandler = ORB::create();
    //describeHandler->compute( img1, keypoints1, descriptors1 );
    //describeHandler->compute( img2, keypoints2, descriptors2 );
    ImageHelper.ComputeBinaryDescriptors(patches1, descriptors1, masks1);     //BOLD
    ImageHelper.ComputeBinaryDescriptors(patches2, descriptors2, masks2);     //BOLD

    //-- Find matches
    vector<DMatch> matches;
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches);
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches, NORM_L2, 0.8);
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches, NORM_HAMMING2, 0.85);
    //ImageHelper.FindMatches(masks1, masks2, matches, NORM_HAMMING2, 0.85);
    ImageHelper.FindMatches(descriptors1, descriptors2, matches, NORM_HAMMING, 0.6);

    //- Draw matches
    Mat imgMatches;
    ImageHelper.DrawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    return 0;
}




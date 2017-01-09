#include "include/utils.h"
#include "include/bold.hpp"
#include "include/helper.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    //-- Read images and resize them
    Mat img1 = imread("img1.jpg", IMREAD_GRAYSCALE );
    Mat img2 = imread("img2.jpg", IMREAD_GRAYSCALE );
    if(!img1.data || !img2.data)
    {
        return 1;
    }

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
    ImageHelper.computePatches(keypoints1, img1, patches1);
    ImageHelper.computePatches(keypoints2, img2, patches2);

    //-- Describe keypoints
    Mat descriptors1, descriptors2;
    Mat masks1, masks2;
    //Ptr<Feature2D> describeHandler = ORB::create();
    //describeHandler->compute( img1, keypoints1, descriptors1 );
    //describeHandler->compute( img2, keypoints2, descriptors2 );
    ImageHelper.computeBinaryDescriptors(patches1, descriptors1, masks1);     //BOLD
    ImageHelper.computeBinaryDescriptors(patches2, descriptors2, masks2);     //BOLD

    //-- Find matches
    vector<DMatch> matches;
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches);
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches, NORM_L2, 0.8);
    //ImageHelper.FindMatches(descriptors1, descriptors2, matches, NORM_HAMMING2, 0.85);
    //ImageHelper.FindMatches(masks1, masks2, matches, NORM_HAMMING2, 0.85);
    ImageHelper.findMatches(descriptors1, descriptors2, matches, NORM_HAMMING, 0.9);

    //- Draw matches
    Mat imgMatches;
    ImageHelper.drawFoundMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    return 0;
}

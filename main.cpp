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
    Mat img2 = imread("s2.jpg", IMREAD_GRAYSCALE );
    resize(img1, img1, Size(600, 800));
    resize(img2, img2, Size(600, 800));

    //-- Detect keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> Detector = SURF::create( minHessian );
    vector<KeyPoint> keypoints1, keypoints2;
    Detector->detect(img1, keypoints1);
    Detector->detect(img2, keypoints2);

    //-- Init Helper
    Helper ImageHelper;

    //-- Get patches of compared images
    vector<Mat> patches1, patches2;
    ImageHelper.ComputePatches(keypoints1, img1, patches1);
    ImageHelper.ComputePatches(keypoints2, img2, patches2);

    //-- Describe keypoints
    Mat descriptors1, descriptors2, masks1, masks2;
    ImageHelper.ComputeBinaryDescriptors(patches1, descriptors1, masks1);
    ImageHelper.ComputeBinaryDescriptors(patches2, descriptors2, masks2);

    //-- Find matches
    std::vector< DMatch > matches;
    ImageHelper.FindBfMatches(descriptors1, descriptors2, matches);

    //- Draw matches
    Mat imgMatches;
    ImageHelper.DrawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    return 0;
    //  /* get descriptors for patch pairs */
    //  for (int i = 0; i < GT_SIZE; i++) {
    //    int id1 = data.gt[i][0];
    //    int id2 = data.gt[i][3];
    //    int label = data.gt[i][1]==data.gt[i][4];

    //    cv::Mat desc_boldL,desc_boldR,mask_boldL,mask_boldR;
    //    bold.compute_patch(data.patchesCV[id1], desc_boldL, mask_boldL) ;
    //    bold.compute_patch(data.patchesCV[id2], desc_boldR, mask_boldR) ;

    ////    cout    << "desc1: " << desc_boldR << endl
    ////            << "desc2: " << desc_boldL << endl
    ////            << "mask_1" << mask_boldL << endl
    ////            << "mask_2" << mask_boldR << endl;

    //    //cv::imshow("window", data.patchesCV[id1]);
    //    //cv::waitKey(0);


    //    //pointers for first descriptors from Mat
    //    uchar *l = desc_boldL.ptr<uchar>(0);
    //    uchar *ml = mask_boldL.ptr<uchar>(0);
    //    uchar *r = desc_boldR.ptr<uchar>(0);
    //    uchar *mr = mask_boldR.ptr<uchar>(0);

    //    //show 8bit descriptors
    //    std::bitset<8>x(*l);
    //    std::bitset<8>y(*r);
    //    cout << "boldL: " << x  << std::endl
    //         << "boldR: " << y << std::endl;
    //    getchar();

    //    /* dfull: no masking */
    //    /* dbold : masking */
    //    int dbold = bold.hampopmaskedLR(l,ml,r,mr);
    //    int dfull = bold.hampop(l,r);

    ////    std::string binary = std::bitset<256>(l).to_string(); //to binary
    ////    cout << binary << endl << endl;

    ////    cout    << "l: : " << desc_boldL << endl
    ////            << "ml: " << ml << endl
    ////            << "r: " << r << endl
    ////            << "ml: " << mr << endl;

    //    getchar();
    //    printf("%d %d %d\n",label,dfull,dbold);
    //  }

    //  /* cleanup */
    //  for (int i=0; i<GT_SIZE; ++i) {
    //    free(data.gt[i]);
    //  }
    //  free(data.gt);
}




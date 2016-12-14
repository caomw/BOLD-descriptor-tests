#include "utils.h"
#include "bold.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    //dataset data;
    //init_dataset(&data,(char*) "dataset/notredame/");

    //-- Read images and resize them
    Mat imgFront = imread("s1.jpg", IMREAD_GRAYSCALE );
    Mat imgSide = imread("s2.jpg", IMREAD_GRAYSCALE );
    resize(imgFront, imgFront, Size(600, 800));
    resize(imgSide, imgSide, Size(600, 800));

    //-- Detect keypoints using SURF Detector
    int minHessian = 500;
    Ptr<SURF> Detector = SURF::create( minHessian );
    vector<KeyPoint> keypointsFront, keypointsSide;
    Detector->detect(imgFront, keypointsFront);
    Detector->detect(imgSide, keypointsSide);

    //-- Get patches of compared images
    Helper ImageHelper;
    vector<Mat> patchesFront = ImageHelper.ComputePatches(keypointsFront, imgFront);
    vector<Mat> patchesSide = ImageHelper.ComputePatches(keypointsSide, imgSide);

    //-- Describe keypoints
    Mat descriptorsFront, masksFront;
    Mat descriptorsSide, masksSide;
    ImageHelper.ComputeBinaryDescriptors(patchesFront, descriptorsFront, masksFront);
    ImageHelper.ComputeBinaryDescriptors(patchesSide, descriptorsSide, masksSide);

//    //cout << descriptorsFront.size() << endl;

    //-- Match descriptors
    BFMatcher matcher(NORM_HAMMING, false);
    vector<DMatch> matches;

    //matcher.match(descriptorsFront, descriptorsSide, matches);  //TODO: add masks
    matcher.match(masksFront, masksSide, matches);  //TODO: add masks

    //-- Draw matches
    Mat imgMatches;
    drawMatches(imgFront, keypointsFront, imgSide, keypointsSide, matches, imgMatches);

    //-- Show detected matches
    imshow("Matches", imgMatches );

    waitKey(0);

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

  return 0;
}

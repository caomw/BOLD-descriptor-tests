#include "Helper.h"

Helper::Helper()
{
    patchSize = CvSize(32, 32);
}

Helper::~Helper() {}

void Helper::ComputePatches(vector<KeyPoint>& keypoints, Mat& img, vector<Mat>& patches)
{
    for(int i = 0; i < keypoints.size(); i++)
    {
        Mat tile = GetPatch(img, keypoints[i]);
        if(tile.rows == patchSize.width && tile.cols == patchSize.height)
        {
            patches.push_back(tile);
        }
    }
}

Mat Helper::GetPatch(Mat img, KeyPoint keypoint)
{
    int row = round(keypoint.pt.x);
    int col = round(keypoint.pt.y);
    Range colRange, rowRange;

    if(row > patchSize.height/2 && col > patchSize.width/2 && col < (img.cols - patchSize.width/2) && row < (img.rows - patchSize.height/2))
    {
        colRange = Range(col - patchSize.width/2, col + patchSize.width/2);
        rowRange = Range(row - patchSize.height/2, row + patchSize.height/2);
        //cout << "rowrange: " << rowRange.start << " " << rowRange.end << endl << "colrange: " << colRange.start << " " << colRange.end << endl << endl;
        return img(rowRange, colRange);
    }
    else
    {
        return img;
    }
}

void Helper::ComputeBinaryDescriptors(vector<Mat>& patches, Mat& descriptors, Mat& masks)
{
    BOLD* Bold = new BOLD();
    for(int i = 0; i < patches.size(); i++)
    {
        Mat tmpDescriptor, tmpMask;

        Bold->compute_patch(patches[i], tmpDescriptor, tmpMask);

        descriptors.push_back(tmpDescriptor);
        masks.push_back(tmpMask);
    }

    delete Bold;
}

void Helper::FindBfMatches(Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches)
{
    //BFMatcher matcher;
    //matcher.match(descriptors1, descriptors2, matches);
    vector<vector<DMatch> > potentialMatches;
    BFMatcher matcher;

    matcher.knnMatch(descriptors1, descriptors2, potentialMatches, 2);  // Find two nearest matches
    for (int i = 0; i < potentialMatches.size(); ++i)
    {
        const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (potentialMatches[i][0].distance < ratio * potentialMatches[i][1].distance)
        {
            matches.push_back(potentialMatches[i][0]);
        }
    }
}

int Helper::Hampopmasked(uchar *a,uchar *ma,uchar *b,uchar *mb)
{
  int distL = 0;
  int distR = 0;
  int nL = 0;
  int nR = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    int xormaskedL = axorb & ma[i]  ;
    int xormaskedR = axorb & mb[i]  ;
    nL += ma[i];
    nR += mb[i];
    distL += __builtin_popcount(xormaskedL);
    distR += __builtin_popcount(xormaskedR);
  }
  float n = nL + nR;
  float wL = nL / n;
  float wR = nR / n;
  return distL*wL + distR*wR;
}

int Helper::Hampop(uchar *a,uchar *b)
{
  int distL = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    distL += __builtin_popcount(axorb);
  }
  return distL;
}

void Helper::SaveKeypointsToFile(string filename, vector<KeyPoint> keypoints)
{
    fstream outputFile;
    outputFile.open(filename.c_str(), ios::out );
    for(size_t i = 0; i < keypoints.size(); i++)
    {
       outputFile << floor(keypoints[i].pt.x) << " " << floor(keypoints[i].pt.y) << endl;
    }
    outputFile.close( );
}

void Helper::DrawMatches(Mat &img1, vector<KeyPoint> &keypoints1, Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches, Mat &imgMatches)
{
    namedWindow("matches", 1);
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    imshow("matches", imgMatches);
    waitKey(0);
}

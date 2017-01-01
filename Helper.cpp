#include "Helper.h"

Helper::Helper()
{
    patchSize = CvSize(32, 32);
}
Helper::~Helper()
{
}

/* masked distance  */
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

/* hamming distance  */
int Helper::Hampop(uchar *a,uchar *b)
{
  int distL = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    distL += __builtin_popcount(axorb);
  }
  return distL;
}

void Helper::FindMatches(vector<myMatch> desca, vector<myMatch> descb, vector<vector<Point> >& results)
{
    //brute force matcher
    for(int i=0;i<desca.size();i++)
    {
        Point p;
        vector<Point> points;
        int index = 0;
        for(int j = 1; j < descb.size();j++)
        {
            //int resultPrevious = Hampop(&desca[i].descValue, &descb[j-1].descValue);
            //int resultCurrent = Hampop(&desca[i].descValue, &descb[j].descValue);
            int resultPrevious = Hampopmasked(&desca[i].descValue, &desca[i].maskValue, &descb[j-1].descValue, &descb[j-1].maskValue);
            int resultCurrent = Hampopmasked(&desca[i].descValue, &desca[i].maskValue, &descb[j].descValue, &descb[j].maskValue);


            //cout << resultCurrent << " " << resultPrevious << endl;

            if(resultCurrent < resultPrevious)
            {
                p.x = descb[j].pt.x;
                p.y = descb[j].pt.y;
                index = j;
            }
        }

        //cout << desca[i].pt << " " << p << endl;
        points.push_back(desca[i].pt);
        points.push_back(p);
        results.push_back(points);
        descb.erase(descb.begin() + index);
    }
}

void Helper::ComputeBinaryDescriptors(vector<myMatch>& patches, string filename, int descNum)
{
    BOLD* Bold = new BOLD(filename, descNum);
    //BOLD* Bold = new BOLD();
    for(int i = 0; i < patches.size(); i++)
    {
        Mat tmpDescriptor, tmpMask;

        Bold->compute_patch(patches[i].patch, tmpDescriptor, tmpMask);
        uchar *desc = tmpDescriptor.ptr<uchar>(0);
        uchar *mask = tmpMask.ptr<uchar>(0);

        //cout << (int)*desc << " " << (int)*mask << endl;

        patches[i].descValue = *desc;
        patches[i].maskValue = *mask;
    }

    delete Bold;
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

vector<myMatch> Helper::ComputePatches(vector<KeyPoint> keypoints, Mat img)
{
//    namedWindow("aaa", WINDOW_NORMAL);
    vector<myMatch> patches;
    for(int i = 0; i < keypoints.size(); i++)
    {
        Mat tile = GetPatch(img, keypoints[i]);
        if(tile.rows == patchSize.width && tile.cols == patchSize.height)
        {
            myMatch m;
            m.patch = tile;
            m.pt = Point(keypoints[i].pt.x, keypoints[i].pt.y);
            patches.push_back(m);
        }
    }
    return patches;
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

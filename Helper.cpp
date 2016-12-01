#include "Helper.h"

Helper::Helper()
{
    patchSize = CvSize(32, 32);
}
Helper::~Helper()
{
}

void Helper::ComputeBinaryDescriptors(vector<Mat> patches, Mat& descriptors, Mat& masks)
{
    BOLD Bold;
    for(int i = 0; i < patches.size(); i++)
    {
        Mat tmpDescriptor, tmpMask;
        Bold.compute_patch(patches[i], tmpDescriptor, tmpMask);

        uchar *desc = tmpDescriptor.ptr<uchar>(0);
        uchar *mask = tmpMask.ptr<uchar>(0);

        descriptors.push_back(*desc);
        masks.push_back(*mask);
    }
}

vector<Mat> Helper::ComputePatches(vector<KeyPoint> keypoints, Mat img)
{
    vector<Mat> patches;
    for(int i = 0; i < keypoints.size(); i++)
    {
        Mat tile = GetPatch(img, keypoints[i]);
        patches.push_back(tile);
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
        //cout << "rowrange: " << rowRange.start << " " << rowRange.end << endl << "colrange: " << colRange.start << colRange.end << endl << endl;
        return img(rowRange, colRange);
    }
    else
    {
        return img;
    }
}

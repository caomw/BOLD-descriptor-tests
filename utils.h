#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <bitset>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <fstream>
#include <cstddef>

/* Dataset */
using namespace cv;

#define GT_SIZE 100000
typedef struct dataset 
{
  int **gt;
  int npatches;
  std::vector<cv::Mat> patchesCV;
} dataset;

void init_dataset(dataset *A,const char *path);

#endif /* _UTILS_H_ */

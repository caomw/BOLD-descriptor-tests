#include "bold.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <bitset>

/* load tests and init 2 rotations for fast affine aprox. (example -20,20) */
BOLD::BOLD(string filename, int descNum)
{
    //bin_tests = (int**) malloc(NROTS * sizeof(int *));    //NROTS = 3, sizeof(int*) = 8 -> allocate 27 bytes of memory (for 3 int pointers)   //--C style
    bin_tests = (int**)new int[NROTS];  //--C++ style
    for (int i = 0; i < NROTS; i++)
    {
        //bin_tests[i] = (int*)malloc(NTESTS*2 * sizeof(int));    //NTESTS*2 = 1524, sizeof(int) = 4 - allocate 6096 bytes of memory (for 1524 int numbers) //--C style,
        bin_tests[i] = new int[descNum*2];   //--C++ style
    }
    std::ifstream file(filename.c_str(), ios::in);
//    file.open("bold.descr");
    //file.open(filename.c_str(), ios::in);

    /* read original tests and set them to rotation 0 */
    for(int j = 0; j < descNum*2; j++ )
    {
        file >> bin_tests[0][j];
    }

    file.close();
    rotations[0] = 20;
    rotations[1] = -20;

    /* compute the rotations offline */
    for (int i = 0; i < descNum*2; i+=2)
    {
        int x1 = bin_tests[0][i] % 32;
        int y1 = bin_tests[0][i] / 32;
        int x2 = bin_tests[0][i+1] % 32;
        int y2 = bin_tests[0][i+1] / 32;
        for (int a = 1; a < NROTS; a++)
        {
            float angdeg = rotations[a-1];
            float angle = angdeg*(float)(CV_PI/180.f);
            float ca = (float)cos(angle);
            float sa = (float)sin(angle);
            int rotx1 = (x1-15)*ca - (y1-15)*sa + 16;
            int roty1 = (x1-15)*sa + (y1-15)*ca + 16;
            int rotx2 = (x2-15)*ca - (y2-15)*sa + 16;
            int roty2 = (x2-15)*sa + (y2-15)*ca + 16;
            bin_tests[a][i] = rotx1 + 32*roty1;
            bin_tests[a][i+1] = rotx2 + 32*roty2;
        }
    }
}

/* bin_tests.size = 3x1536
 0x0 -> from file
 1x0 -> from file * +rot
 2x0 -> from file * -rot
*/

BOLD::~BOLD(void)
{
  /* free the tests */
  for (int i = 0; i < NROTS; i++)
  {
    free(bin_tests[i]);
  }
  free(bin_tests);
}

//returns descriptor 1x64 0-255
void BOLD::compute_patch(cv::Mat img, cv::Mat& descr,cv::Mat& masks)
{
  /* init cv mats */
  int nkeypoints = 1;
  descr.create(nkeypoints, DIMS/8, CV_8U); //descriptor -> 1x64x[0-255]
  masks.create(nkeypoints, DIMS/8, CV_8U); //mask -> 1x64x[0-255]

  /* apply box filter */
  cv::Mat patch;
  boxFilter(img, patch, img.depth(), cv::Size(5,5), cv::Point(-1,-1), true, cv::BORDER_REFLECT); //blur patch

  /* get test and mask results  */
  int k =0;
  uchar* dsc = descr.ptr<uchar>(k);
  uchar* msk = masks.ptr<uchar>(k);
  uchar *smoothed = patch.data; //pointer to the first row of mat patch -> 111dec

  //pointer for first tests in patch
  int* tests = bin_tests[0];
  int* r0 = bin_tests[1];
  int* r1 = bin_tests[2];

  //returned values
  unsigned int val = 0;
  unsigned int var = 0;
  int idx = 0;
  int j=0;
  int bit;
  int tdes,tvar;

  //compare all tests by collumns
  for (int i = 0; i < DIMS; i++, j+=2)
  {
//        std::bitset<8>x(tests[j]);
//        std::cout << tests[j] << std::endl;
//        getchar();


        bit = i%8;
        int temp_var = 0;

        //binary tree for all tests in patch
        tdes = (smoothed[tests[j]] < smoothed[tests[j+1]]); // 0 or 1: no rot
        temp_var += (smoothed[r0[j]] < smoothed[r0[j+1]])^tdes; // 0 or 1: + rot
        temp_var += (smoothed[r1[j]] < smoothed[r1[j+1]])^tdes; // 0 or 1: - rot
//        std::cout << "tdes: " << tdes << std::endl
//                 << "smoothed[tests[j]]: " << smoothed[tests[j]] << std::endl
//                 << "smoothed[tests[j+1]]" << smoothed[tests[j+1]] << std::endl;
//        getchar();

        /* tvar-> 0 not stable --------  tvar-> 1 stable */
        tvar = (temp_var == 0) ;
        if (bit==0)
        {
            val = tdes;
            var = tvar;
        }
        else
        {
            val |= tdes << bit;
            var |= tvar << bit;
        }
        if (bit==7)
        {
            dsc[idx] = val;
            msk[idx] = var;
            val = 0;
            var = 0;
            idx++;
        }
    }

//  std::bitset<8>x(*dsc);
//  std::cout << x  << std::endl;
}

/* masked distance  */
int BOLD::hampopmaskedLR(uchar *a,uchar *ma,uchar *b,uchar *mb)
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
int BOLD::hampop(uchar *a,uchar *b)
{
  int distL = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    distL += __builtin_popcount(axorb);
  }
  return distL;
}

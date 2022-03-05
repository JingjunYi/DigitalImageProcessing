// DIP.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define PI 3.1415926

using namespace std;  //使用cout输出方式
using namespace cv;   // 省去函数前面加cv::的必要性

Mat getsingleband(Mat& M, int band);
Mat graylinechange_RGB(Mat& M, double k, double d, int band);
Mat graylinechange_Gray(Mat& M, double k, double d);
Mat highpassfilter_LE(Mat& M);
Mat lowpassfilter_A(Mat& M);
Mat mediumfilter(Mat& M);
Mat translation(Mat& M, int direction, int distance);
Mat resizefornearest(Mat& M, float rsize, float csize);
Mat rotation_CW(Mat& M, float degree);
Mat rotation_cv(Mat& M,float degree);
Mat line_fakecolor_RGB(Mat& M);
Mat colorbalance_RGB(Mat& M);
Mat templatematching_SAD(Mat& src, Mat& mask);
Mat makemask(Mat& src, Point corner, int mheight, int mwidth);

int main()
{
    FILE* fp;

    if (fopen_s(&fp, "20180620-tianjin-2076x2816x3BIP", "rb") != 0)
    {
        printf("cannot open file for read\n");
        waitKey();//opencv中常用waitKey(n),n<=0表示一直
        exit(0);
    }


    int cols = 2076;
    int rows = 2816;
    int bands = 3;// 波段数
    int pixels = cols * rows * bands;

    unsigned char* data = new uchar[pixels * sizeof(uchar)];
    fread(data, sizeof(uchar), pixels, fp);
    fclose(fp);

    //Mat M(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    //unsigned char* ptr = M.data;

    //for (int i = 0; i < rows; i++) {
    //    for (int j = 0; j < cols; j++) 
    //    {
    //        for (int k = 0; k < 3; k++)
    //        {
    //            ptr[(i * cols + j) * 3 + k] = data[(i * cols + j) * bands + k];//BIP
    //            //ptr[(i * cols + j) * 3 + k] = data[i * cols * bands + (k + 3) * cols + j];//BIL
    //            //ptr[(i * cols + j) * 3 + k] = data[(i + rows * k) * cols + j];//BSQ
    //        }
    //    }
    //}


    //namedWindow("image", 1);//1为按图片大小显示，0为跟据窗口大小调整
    //imshow("image", M);  // 显示图片 
    //waitKey();
    //imwrite("pic.bmp", M); // 存为bmp格式图片





    //Compulsory Algorithm One

    //linear transform of gray scale
    /*Mat M1 = graylinechange_RGB(M, 2, 1, 1);
    namedWindow("image", 1);
    imshow("image", M1);
    waitKey();
    Mat M2 = getsingleband(M, 1);
    imshow("singleband", M2);
    waitKey();
    imwrite("singleband.bmp", M2);
    Mat M3 = graylinechange_Gray(M2, 2, 1);
    namedWindow("image", 1);
    imshow("image", M3);
    waitKey();
    imwrite("graylinechange.bmp", M1);*/
    /*Mat M_1 = imread("ik_beijing_p.bmp", 0);
    namedWindow("image", 1);
    imshow("image", M_1);
    waitKey();
    Mat M4 = graylinechange_Gray(M_1,2,-1);
    imshow("graylinechange", M4);
    waitKey();
    imwrite("graylinechange_beijing.bmp", M4);*/



    //Compulsory Algorithm Two

    //highpass filter
    /*Mat M11 = getsingleband(M, 2);
    Mat M12 = highpassfilter_LE(M11);
    namedWindow("image", 1);
    imshow("image", M11);
    waitKey();
    imshow("highpassfilter", M12);
    waitKey();
    imwrite("highpassfilter.bmp", M12);*/
    /*Mat m11 = imread("ik_beijing_p.bmp", 0);
    Mat m12 = highpassfilter_LE(m11);
    namedWindow("image", 1);
    imshow("image", m11);
    waitKey();
    imshow("highpassfilter", m12);
    waitKey();
    imwrite("highpassfilter_beijing.bmp", m12);*/

    //lowpass filter
    /*Mat M21 = getsingleband(M, 2);
    Mat M22 = lowpassfilter_A(M21);
    namedWindow("image", 1);
    imshow("image", M21);
    waitKey();
    imshow("lowpassfilter", M22);
    waitKey();
    imwrite("lowpassfilter.bmp", M22);*/
    /*Mat m21 = imread("ik_beijing_p.bmp", 0);
    Mat m22 = lowpassfilter_A(m21);
    namedWindow("image", 1);
    imshow("image", m21);
    waitKey();
    imshow("highpassfilter", m22);
    waitKey();
    imwrite("lowpassfilter_beijing.bmp", m22);*/

    //medium filter
    /*Mat M31 = getsingleband(M, 3);
    Mat M32 = lowpassfilter_A(M31);
    namedWindow("image", 1);
    imshow("image", M31);
    waitKey();
    imshow("mediumfilter", M31);
    waitKey();
    imwrite("mediumfilter.bmp", M32);*/
    /*Mat m31 = imread("ik_beijing_p.bmp", 0);
    Mat m32 = lowpassfilter_A(m31);
    namedWindow("image", 1);
    imshow("image", m31);
    waitKey();
    imshow("mediumfilter", m32);
    waitKey();
    imwrite("mediumfilter_beijing.bmp", m32);*/



    //Optional Algorithm One - Issue One

    //image translation
    /*Mat M41 = imread("ik_beijing_p.bmp", 0);
    Mat M42 = translation(M41, 1, 50);
    namedWindow("image", 1);
    imshow("image", M41);
    waitKey();
    imshow("translation", M42);
    waitKey();
    imwrite("translation.bmp", M42);*/

    //image zoom
    /*Mat M51 = imread("ik_beijing_p.bmp", 0);
    Mat M52 = resizefornearest(M51, 1.5, 1.5);
    namedWindow("image", 1);
    imshow("image", M51);
    waitKey();
    imshow("zoom", M52);
    waitKey();
    imwrite("zoom.bmp", M52);*/

    //image rotation
    /*Mat M61 = imread("ik_beijing_p.bmp", 0);
    Mat M62 = rotation_CW(M61, 50);
    Mat M63 = rotation_cv(M61, 50);
    imshow("image", M61);
    waitKey();
    imshow("rotation", M62);
    waitKey();
    imshow("ratation_cv",M63);
    waitKey();
    imwrite("rotation.bmp", M62);
    imwrite("rotation_cv.bmp", M63);*/



    //Optional Algorithm Two - Issue Four

    //fake color enhancement
    /*Mat M71 = imread("ik_beijing_p.bmp", 0);
    Mat M72 = line_fakecolor_RGB(M71);
    namedWindow("image", 1);
    imshow("image", M71);
    waitKey();
    imshow("fakecolor", M72);
    waitKey();
    imwrite("fakecolor1.bmp", M72);*/

    //Optional Algorithm Three - Issue Five

    //RGB color balance
    /*Mat M81 = imread("ik_beijing_c.bmp",1);
    Mat M82 = colorbalance_RGB(M81);
    namedWindow("image", 1);
    imshow("image", M81);
    waitKey();
    imshow("colorbalance", M82);
    waitKey();
    imwrite("colorbalance.bmp", M82);*/



    //Template Matching SAD
    Mat src = imread("1/SmokingMan_Face.bmp", 0);
    Mat mask = imread("1/Eye.bmp", 0);
    Mat match = templatematching_SAD(src, mask);
    imshow("match", match);
    waitKey();
    imwrite("match.bmp", match);

    /*Mat src = imread("1/SmokingMan.bmp", 0);
    Mat mask1 = makemask(src, Point(380, 168), 500, 500);*/
    /*imshow("mask1", mask1);
    waitKey();
    imwrite("1/mask1.bmp", mask1);*/
    /*Mat mask2 = makemask(mask1, Point(120, 18), 160, 200);*/
    /*imshow("mask2", mask2);
    waitKey();
    imwrite("1/mask2.bmp", mask2);*/
    /*Mat mask3 = makemask(mask2, Point(40, 65), 25, 30);
    imshow("mask3", mask3);
    waitKey();
    imwrite("1/mask3.bmp",mask3);*/
    /*Mat src = imread("1/mask2.bmp", 0);
    Mat mask = imread("1/mask3.bmp", 0);
    Mat match = templatematching_SAD(src, mask);
    imshow("match", match);
    waitKey();
    imwrite("1/match.bmp", match);*/

    return 0;
}


Mat getsingleband(Mat& M, int band)
{
    Mat N = Mat(M.rows, M.cols, CV_8UC1);
    unsigned char* ptr = N.data;
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            ptr[i * M.cols + j] = M.data[(i * M.cols + j) * 3 + (band + 1)];
        }
    }
    return N;
}

Mat graylinechange_RGB(Mat& M, double k, double d, int band)
{
    Mat N = Mat(M.rows, M.cols, CV_8UC3);
    unsigned char* ptr = N.data;
    for (int i = 0; i < M.rows; i++) 
    {
        for (int j = 0; j < M.cols; j++) 
        {
            for (int k = 0; k < 3; k++)
            {
                ptr[(i * M.cols + j) * 3 + k] = M.data[(i * M.cols + j) * 3 + k];//BIP
            }
        }
    }
    //for any band
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            if (k * M.data[(i * M.cols + j) * 3 + (band - 1)] + d > 255)
            {
                ptr[(i * M.cols + j) * 3 + (band - 1)] = 255;
            }
            else if (k * M.data[(i * M.cols + j) * 3 + (band - 1)] + d < 0)
            {
                ptr[(i * M.cols + j) * 3 + (band - 1)] = 0;
            }
            else
            {
                ptr[(i * M.cols + j) * 3 + (band - 1)] = int(k * M.data[(i * M.cols + j) * 3 + (band - 1)] + d);
            }
        }
    }
    return N;
}
Mat graylinechange_Gray(Mat& M, double k, double d)
{
    Mat N = Mat(M.rows, M.cols, CV_8UC1);
    unsigned char* ptr = N.data;
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            ptr[i * M.cols + j] = M.data[i * M.cols + j];
        }
    }
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            if (k * M.data[i * M.cols + j] + d > 255)
            {
                ptr[i * M.cols + j] = 255;
            }
            else if (k * M.data[i * M.cols + j] + d < 0)
            {
                ptr[i * M.cols + j] = 0;
            }
            else
            {
                ptr[i * M.cols + j] = int(k * M.data[i * M.cols + j] + d);
            }
        }
    }
    return N;
}

Mat highpassfilter_LE(Mat& M)//for single band(gray)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC1);
    M.copyTo(N);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i == 0 || j == 0)//keep four sides of the matrix
            {
                N.data[i * width + j] = M.data[i * width + j];
            }
            else//conv with Laplacian enhancement operator{0,-1,0;-1,5,-1;0,-1,0}
            {
                int m0 = M.data[i * width + j];
                int m1 = M.data[i * (width - 1) + j];
                int m2 = M.data[i * width + j - 1];
                int m3 = M.data[i * width + j + 1];
                int m4 = M.data[i * (width + 1) + j];
                if (5 * m0 - m1 - m2 - m3 - m4 > 255)
                {
                    N.data[i * width + j] = 255;
                }
                else if (5 * m0 - m1 - m2 - m3 - m4 < 0)
                {
                    N.data[i * width + j] = 0;
                }
                else
                {
                    N.data[i * width + j] = 5 * m0 - m1 - m2 - m3 - m4;
                }
            }
        }
    }
    return N;
}

Mat lowpassfilter_A(Mat& M)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC1);
    M.copyTo(N);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i == 0 || j == 0)//keep four sides of the matrix
            {
                N.data[i * width + j] = M.data[i * width + j];
            }
            else//conv with average operator{1/16,2/16,1/16;2/16,4/16,2/16;1/16,2/16,1/16}
            {
                int m0 = M.data[i * width + j];
                int m1 = M.data[i * (width - 1) + j];
                int m2 = M.data[i * width + j - 1];
                int m3 = M.data[i * width + j + 1];
                int m4 = M.data[i * (width + 1) + j];
                int m5 = M.data[i * (width - 1) + j - 1];
                int m6 = M.data[i * (width - 1) + j + 1];
                int m7 = M.data[i * (width + 1) + j - 1];
                int m8 = M.data[i * (width + 1) + j + 1];
                if (int((4 * m0 + 2 * m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + m6 + m7 + m8) / 16) > 255)
                {
                    N.data[i * width + j] = 255;
                }
                else if (int((4 * m0 + 2 * m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + m6 + m7 + m8) / 16) < 0)
                {
                    N.data[i * width + j] = 0;
                }
                else
                {
                    N.data[i * width + j] = int((4 * m0 + 2 * m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + m6 + m7 + m8) / 16);
                }
            }
        }
    }
    return N;
}

Mat mediumfilter(Mat& M)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC1);
    M.copyTo(N);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i == 0 || j == 0)
            {
                N.data[i * width + j] = M.data[i * width + j];
            }
            else//medium value of 3x3 domain
            {
                int a[9] = {};
                a[0] = M.data[i * width + j];
                a[1] = M.data[i * (width - 1) + j];
                a[2] = M.data[i * width + j - 1];
                a[3] = M.data[i * width + j + 1];
                a[4] = M.data[i * (width + 1) + j];
                a[5] = M.data[i * (width - 1) + j - 1];
                a[6] = M.data[i * (width - 1) + j + 1];
                a[7] = M.data[i * (width + 1) + j - 1];
                a[8] = M.data[i * (width + 1) + j + 1];
                sort(a,a+9);
                N.data[i * width + j] = a[4];
            }
        }
    }
        return N;
}

Mat translation(Mat& M, int direction, int distance)//for single band(gray)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC1);
    M.copyTo(N);
    if (direction == 1)//up
    {
        if (distance > 0 && distance < height)
        {
            for (int i = height - distance; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    N.data[i * width + j] = 0;
                }
            }
            for (int i = 0; i < height - distance; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    N.data[i * width + j] = M.data[(i + distance) * width + j];
                }
            }
        }
    }
    else if (direction == 2)//right
    {
        if (distance > 0 && distance < width)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < distance; j++)
                {
                    N.data[i * width + j] = 0;
                }
            }
            for (int i = 0; i < height; i++)
            {
                for (int j = distance; j < width; j++)
                {
                    N.data[i * width + j] = M.data[i * width + j - distance];
                }
            }
        }
    }
    else if (direction == 3)//down
    {
        if (distance > 0)
        {
            if (distance > 0 && distance < height)
            {
                for (int i = 0; i < distance; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        N.data[i * width + j] = 0;
                    }
                }
                for (int i = distance; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        N.data[i * width + j] = M.data[(i - distance) * width + j];
                    }
                }
            }
        }
    }
    else if (direction == 4)//left
    {
        if (distance > 0 && distance < width)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = width - distance; j < width; j++)
                {
                    N.data[i * width + j] = 0;
                }
            }
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width - distance; j++)
                {
                    N.data[i * width + j] = M.data[i * width + j + distance];
                }
            }
        }
    }
    return N;
}

Mat resizefornearest(Mat& M, float rsize, float csize)
{
    int height = int(M.rows * rsize);
    int width = int(M.cols * csize);
    Mat N = Mat(height, width, CV_8UC1, Scalar::all(0));
    for (int i = 0; i < height; i++)
    {
        int srcr = cvFloor(i / rsize);
        srcr = min(srcr, M.rows - 1);
        for (int j = 0; j < width; j++)
        {
            int srcc = cvFloor(j / csize);
            srcc = min(srcc, M.cols - 1);
            N.data[i * width + j] = M.data[srcr * M.cols + srcc];
        }
    }
    return N;
}

Mat rotation_CW(Mat& M, float degree)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC1);
    double sind = sin(degree);
    double cosd = cos(degree);
    double xr = M.cols / 2;
    double yr = M.rows / 2;
    double constx = -xr * cosd + yr * sind + xr;
    double consty = -yr * cosd - xr * sind + yr;
    double x1;
    double y1;
    int x0;
    int y0;
    for (int y = 0; y < height; y++)
    {
        x1 = -y * sind - cosd + constx;
        y1 = y * cosd - sind + consty;
        for (int x = 0; x < width; x++)
        {
            x1 += cosd;
            y1 += sind;
            x0 = int(x1);
            y0 = int(y1);
            if (x0 > 0 && x0 < width - 1 && y0>0 && y0 < height - 1)
            {
                N.data[y * width + x] = M.data[y0 * width + x0];
            }
            else
            {
                N.data[y * width + x] = 0;
            }
        }
    }
    return N;
}
Mat rotation_cv(Mat& M, float degree)//without insert value
{
    int height = M.rows;
    int width = M.cols;
    Mat N;
    float rad = (float)(degree / 180) * PI;
    int maxborder = max(height, width) * 1.414;//find max matrix to cover all situation
    int dx = (maxborder - width) / 2;
    int dy = (maxborder - height) / 2;
    copyMakeBorder(M, N, dy, dy, dx, dx, BORDER_CONSTANT);//broaden the matrix
    Point2f center = Point2f(N.cols / 2, N.rows / 2);//rotation center point
    Mat rotation_matrix = getRotationMatrix2D(center, degree, 1);
    warpAffine(N, N, rotation_matrix, N.size());//change axis
    float sinval = abs(sin(rad));
    float cosval = abs(cos(rad));
    Size Nsize = Size((width * cosval + height * sinval), (width * sinval + height * cosval));//calculate the current size of rotated matrix
    int cx = (N.cols - Nsize.width) / 2;
    int cy = (N.rows - Nsize.height) / 2;
    Rect rect = Rect(cx, cy, Nsize.width, Nsize.height);
    N = Mat(N, rect);//polish the matrix
    return N;

}

Mat line_fakecolor_RGB(Mat& M)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat::zeros(height, width, CV_8UC3);
    Mat N1 = graylinechange_Gray(M, 0.8,88); 
    Mat N2 = graylinechange_Gray(M, 1.88,-188);
    Mat N3 = graylinechange_Gray(M, 8.88,-888);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (k == 0)
                {
                    N.data[(i * width + j) * 3 + k] = N1.data[i * width + j];
                }
                else if (k == 1)
                {
                    N.data[(i * width + j) * 3 + k] = N2.data[i * width + j];
                }
                else if (k == 2)
                {
                    N.data[(i * width + j) * 3 + k] = N3.data[i * width + j];
                }
            }
        }
    }
    return N;
}

Mat colorbalance_RGB(Mat& M)
{
    int height = M.rows;
    int width = M.cols;
    Mat N = Mat(height, width, CV_8UC3);
    double y;
    double ymax = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            y = 0.299 * M.data[(i * width + j) * 3 + 2] + 0.587 * M.data[(i * width + j) * 3 + 1] + 0.144 * M.data[(i * width + j) * 3 + 0];
            if (y > ymax)
            {
                ymax = y;
            }
        }
    }
    double aR = 0;
    int nR = 0;
    double aG = 0;
    int nG = 0;
    double aB = 0;
    int nB = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if ((0.299 * M.data[(i * width + j) * 3 + 2] + 0.587 * M.data[(i * width + j) * 3 + 1] + 0.144 * M.data[(i * width + j) * 3 + 0]) <= 0.95 * ymax)
            {
                aR += M.data[(i * width + j) * 3 + 2];
                nR++;
                aG += M.data[(i * width + j) * 3 + 1];
                nG++;
                aB += M.data[(i * width + j) * 3 + 0];
                nB++;
            }
        }
    }
    aR = (double)(aR / nR);
    aG = (double)(aG / nG);
    aB = (double)(aB / nB);
    double a[3] = {};
    a[0] = aR;
    a[1] = aG;
    a[2] = aB;
    sort(a, a + 3);
    double Bmax = a[2];
    double KR = Bmax / aR;
    double KG = Bmax / aG;
    double KB = Bmax / aB;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            N.data[(i * width + j) * 3 + 2] = int(M.data[(i * width + j) * 3 + 2] * KR);
            N.data[(i * width + j) * 3 + 1] = int(M.data[(i * width + j) * 3 + 1] * KG);
            N.data[(i * width + j) * 3 + 0] = int(M.data[(i * width + j) * 3 + 0] * KB);
        }
    }
    return N;
}

Mat templatematching_SAD(Mat& src, Mat& mask)
{
    int srch = src.rows;
    int srcw = src.cols;
    int maskh = mask.rows;
    int maskw = mask.cols;
    Mat temp = Mat::zeros(maskh, maskw, CV_8UC1);
    Mat sad = Mat::zeros(srch - maskh, srcw - maskw, CV_8UC1);
    Mat m = Mat::zeros(maskh, maskw, CV_8UC1);
    Mat N = src;
    for (int i = 0; i < srch - maskh; i++)
    {
        for (int j = 0; j < srcw - maskw; j++)
        {
            for (int x = 0; x < maskh; x++)
            {
                for (int y = 0; y < maskw; y++)
                {
                    temp.data[x * maskw + y] = N.data[(i + x) * srcw + (j + y)];
                    m.data[x * maskw + y] = mask.data[x * maskw + y] - temp.data[x * maskw + y];
                }
            }
            for (int x = 0; x < maskh; x++)
            {
                for (int y = 0; y < maskw; y++)
                {
                    sad.data[i * (srcw - maskw) + j] += abs(m.data[x * maskw + y]);
                }
            }
        }
    }

    int di = 0;
    int dj = 0;
    uchar d = CHAR_MAX;
    for (int i = 0; i < srch - maskh; i++)
    {
        for (int j = 0; j < srcw - maskw; j++)
        {
            if (sad.data[i * (srcw - maskw) + j] < d)
            {
                d = sad.data[i * (srcw - maskw) + j];
                di = i;
                dj = j;
            }
        }
    }

    for (int i = 0; i < srch - maskh; i++)
    {
        for (int j = 0; j < srcw - maskw; j++)
        {
            if (sad.data[i * (srcw - maskw) + j] == d)
            {
                Point center = Point(dj, di);
                //cout << di << endl;
                //cout << dj << endl;
                Point centerc = Point(dj + maskw - 1, di + maskh - 1);
                rectangle(N, center, centerc, Scalar(255, 255, 255), 2);
            }
        }
    }
    
    return N;
}
Mat makemask(Mat& src, Point corner, int mheight, int mwidth)
{
    int height = src.rows;
    int width = src.cols;
    Mat N = Mat::zeros(mheight, mwidth, CV_8UC1);
    for (int i = 0; i < mheight; i++)
    {
        for (int j = 0; j < mwidth; j++)
        {
            N.data[i * mwidth + j] = src.data[(i + corner.y) * width + (j + corner.x)];
        }
    }
    return N;
}
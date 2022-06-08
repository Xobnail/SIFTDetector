#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>

#define maskSize 3
#define dbl CV_64FC1
#define uchr CV_8UC1
#define shrt CV_16SC1

using namespace cv;
using namespace std;

void countXD(Mat *dog, double *XD, double *detH, double *DXD, int *d, int i, int j, int layer)
{
    double dx = 0.0;
    double dy = 0.0;
    double ds = 0.0;
    double dxx = 0.0;
    double dxy = 0.0;
    double dxs = 0.0;
    double dyx = 0.0;
    double dyy = 0.0;
    double dys = 0.0;
    double dsx = 0.0;
    double dsy = 0.0;
    double dss = 0.0;
    double B[3][3];
    double center;

    center = dog[layer].at<double>(i, j);
    dx = dog[layer].at<double>(i, j + 1) - dog[layer].at<double>(i, j - 1);
    dy = dog[layer].at<double>(i + 1, j) - dog[layer].at<double>(i - 1, j);
    ds = dog[layer + 1].at<double>(i, j) - dog[layer - 1].at<double>(i, j);
    dxx = dog[layer].at<double>(i, j + 1) + dog[layer].at<double>(i, j - 1) - 2.0 * center;
    dxy = (dog[layer].at<double>(i + 1, j + 1) - dog[layer].at<double>(i + 1, j - 1) - dog[layer].at<double>(i - 1, j + 1) + dog[layer].at<double>(i - 1, j - 1)) / 4.0;
    dxs = (dog[layer + 1].at<double>(i, j + 1) - dog[layer + 1].at<double>(i, j - 1) - dog[layer - 1].at<double>(i, j + 1) + dog[layer - 1].at<double>(i, j - 1)) / 4.0;
    dyx = (dog[layer].at<double>(i + 1, j + 1) - dog[layer].at<double>(i + 1, j - 1) - dog[layer].at<double>(i - 1, j + 1) + dog[layer].at<double>(i - 1, j - 1)) / 4.0;
    dyy = dog[layer].at<double>(i + 1, j) + dog[layer].at<double>(i - 1, j) - 2.0 * center;
    dys = (dog[layer + 1].at<double>(i + 1, j) - dog[layer + 1].at<double>(i - 1, j) - dog[layer - 1].at<double>(i + 1, j) + dog[layer - 1].at<double>(i - 1, j)) / 4.0;
    dsx = (dog[layer + 1].at<double>(i, j + 1) - dog[layer + 1].at<double>(i, j - 1) - dog[layer - 1].at<double>(i, j + 1) + dog[layer - 1].at<double>(i, j - 1)) / 4.0;
    dsy = (dog[layer + 1].at<double>(i + 1, j) - dog[layer + 1].at<double>(i - 1, j) - dog[layer - 1].at<double>(i + 1, j) + dog[layer - 1].at<double>(i - 1, j)) / 4.0;
    dss = dog[layer + 1].at<double>(i, j) + dog[layer - 1].at<double>(i, j) - 2.0 * center;

    *detH = (dxx * dyy * dss) + (dxy * dys * dsx) + (dxs * dyx * dsy) - (dsx * dyy * dxs) - (dsy * dys * dxx) - (dss * dyx * dxy);

    B[0][0] = (dyy * dss) - (dsy * dys);
    B[1][0] = (dsx * dys) - (dyx * dss);
    B[2][0] = (dyx * dsy) - (dsx * dyy);
    B[0][1] = (dsy * dxs) - (dxy * dss);
    B[1][1] = (dxx * dss) - (dsx * dxs);
    B[2][1] = (dsx * dxy) - (dxx * dsy);
    B[0][2] = (dxy * dys) - (dyy * dxs);
    B[1][2] = (dyx * dxs) - (dxx * dys);
    B[2][2] = (dxx * dyy) - (dyx * dxy);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            B[i][j] = -B[i][j] / *detH;
        }
    }

    XD[0] = B[0][0] * dx + B[0][1] * dy + B[0][2] * ds;
    XD[1] = B[1][0] * dx + B[1][1] * dy + B[1][2] * ds;
    XD[2] = B[2][0] * dx + B[2][1] * dy + B[2][2] * ds;

    for (int i = 0; i < 3; i++)
    {
        if (XD[i] > 0.5)
            d[i] = 1;
        else if (XD[i] < -0.5)
            d[i] = -1;
        else
            d[i] = 0;
    }

    *DXD = dog[layer].at<double>(i, j) + 0.5 * (XD[0] * dx + XD[1] * dy + XD[2] * ds);
}

Mat Decrease(Mat src)
{
    Mat Decreased = Mat(src.rows / 2, src.cols / 2, dbl);
    for (int j = 0, c = 0; c < Decreased.cols; j += 2, c++)
    {
        for (int i = 0, r = 0; r < Decreased.rows; i += 2, r++)
        {
            Decreased.at<double>(r, c) = src.at<double>(i, j);
        }
    }
    return Decreased;
}

void findExtrema(Mat *extrema, int exIndex, Mat *dog, int cols, int rows, int layer)
{
    double center;
    for (int j = 1; j < cols - 2; j++)
    {
        for (int i = 1; i < rows - 2; i++)
        {
            center = dog[layer].at<double>(i, j);
            if (center > dog[layer].at<double>(i + 1, j + 1) &&
                    center > dog[layer].at<double>(i + 1, j) &&
                    center > dog[layer].at<double>(i, j + 1) &&
                    center > dog[layer].at<double>(i - 1, j - 1) &&
                    center > dog[layer].at<double>(i - 1, j) &&
                    center > dog[layer].at<double>(i, j - 1) &&
                    center > dog[layer].at<double>(i + 1, j - 1) &&
                    center > dog[layer].at<double>(i - 1, j + 1) &&
                    center > dog[layer + 1].at<double>(i + 1, j + 1) &&
                    center > dog[layer + 1].at<double>(i + 1, j) &&
                    center > dog[layer + 1].at<double>(i, j + 1) &&
                    center > dog[layer + 1].at<double>(i - 1, j - 1) &&
                    center > dog[layer + 1].at<double>(i - 1, j) &&
                    center > dog[layer + 1].at<double>(i, j - 1) &&
                    center > dog[layer + 1].at<double>(i + 1, j - 1) &&
                    center > dog[layer + 1].at<double>(i - 1, j + 1) &&
                    center > dog[layer + 1].at<double>(i, j) &&
                    center > dog[layer - 1].at<double>(i + 1, j + 1) &&
                    center > dog[layer - 1].at<double>(i + 1, j) &&
                    center > dog[layer - 1].at<double>(i, j + 1) &&
                    center > dog[layer - 1].at<double>(i - 1, j - 1) &&
                    center > dog[layer - 1].at<double>(i - 1, j) &&
                    center > dog[layer - 1].at<double>(i, j - 1) &&
                    center > dog[layer - 1].at<double>(i + 1, j - 1) &&
                    center > dog[layer - 1].at<double>(i - 1, j + 1) &&
                    center > dog[layer - 1].at<double>(i, j) 
                    ||
                    center < dog[layer].at<double>(i + 1, j + 1) &&
                    center < dog[layer].at<double>(i + 1, j) &&
                    center < dog[layer].at<double>(i, j + 1) &&
                    center < dog[layer].at<double>(i - 1, j - 1) &&
                    center < dog[layer].at<double>(i - 1, j) &&
                    center < dog[layer].at<double>(i, j - 1) &&
                    center < dog[layer].at<double>(i + 1, j - 1) &&
                    center < dog[layer].at<double>(i - 1, j + 1) &&
                    center < dog[layer + 1].at<double>(i + 1, j + 1) &&
                    center < dog[layer + 1].at<double>(i + 1, j) &&
                    center < dog[layer + 1].at<double>(i, j + 1) &&
                    center < dog[layer + 1].at<double>(i - 1, j - 1) &&
                    center < dog[layer + 1].at<double>(i - 1, j) &&
                    center < dog[layer + 1].at<double>(i, j - 1) &&
                    center < dog[layer + 1].at<double>(i + 1, j - 1) &&
                    center < dog[layer + 1].at<double>(i - 1, j + 1) &&
                    center < dog[layer + 1].at<double>(i, j) &&
                    center < dog[layer - 1].at<double>(i + 1, j + 1) &&
                    center < dog[layer - 1].at<double>(i + 1, j) &&
                    center < dog[layer - 1].at<double>(i, j + 1) &&
                    center < dog[layer - 1].at<double>(i - 1, j - 1) &&
                    center < dog[layer - 1].at<double>(i - 1, j) &&
                    center < dog[layer - 1].at<double>(i, j - 1) &&
                    center < dog[layer - 1].at<double>(i + 1, j - 1) &&
                    center < dog[layer - 1].at<double>(i - 1, j + 1) &&
                    center < dog[layer - 1].at<double>(i, j))
                extrema[exIndex].at<short>(i, j) = 255;
            else
                extrema[exIndex].at<short>(i, j) = 0;
        }
    }
}

void Clarify(Mat *extrema, int exIndex, Mat *dog, int cols, int rows, int layer, int height)
{
    for (int j = 1; j < cols - 1; j++)
    {
        for (int i = 1; i < rows - 1; i++)
        {
            if (extrema[exIndex].at<short>(i, j) == 255)
            {
                double detH = 0.0;
                double XD[3];
                double DXD = 0.0;

                int d[3] = {0, 0, 0};
                int di = 0;
                int dj = 0;
                int dh = 0;
                countXD(dog, XD, &detH, &DXD, d, i, j, layer);

                if (detH == 0.0)
                    extrema[exIndex].at<short>(i, j) = 0;

                if (d[0] != 0 || d[1] != 0 || d[2] != 0)
                {
                    di += d[1];
                    dj += d[0];
                    dh += d[2];
                    for (int count = 0;
                         (d[0] != 0 || d[1] != 0 || d[2] != 0) &&
                         ((i + di) > 1 && (i + di) < rows - 1 && (j + dj) > 1 && (j + dj) < cols - 1 && (layer + dh) > 1 && (layer + dh) < height - 1) &&
                         ((i + d[1]) > 1 && (i + d[1]) < rows - 1 && (j + d[0]) > 1 && (j + d[0]) < cols - 1 && (layer + d[2]) > 1 && (layer + d[2]) < height - 1) &&
                         (i != i + di || j != j + dj || layer != layer + dh) &&
                         (count < 100);
                         count++)
                    {
                        countXD(dog, XD, &detH, &DXD, d, i + d[1], j + d[0], layer + d[2]);
                        di += d[1];
                        dj += d[0];
                        dh += d[2];
                    }
                    if (d[0] == 0 && d[1] == 0 && d[2] == 0)
                    {
                        extrema[exIndex].at<short>(i, j) = 0;
                        extrema[exIndex + dh].at<short>(i + di, j + dj) = 255;
                    }
                    else
                        extrema[exIndex].at<short>(i, j) = 0;
                }
            }
        }
    }
}

void SmallCheck(Mat *extrema, int exIndex, Mat *dog, int cols, int rows, int layer, int height)
{
    for (int j = 1; j < cols - 1; j++)
    {
        for (int i = 1; i < rows - 1; i++)
        {
            if (extrema[exIndex].at<short>(i, j) == 255)
            {
                double detH = 0.0;
                double XD[3];
                double DXD = 0.0;
                int d[3] = {0, 0, 0};
                int di = 0;
                int dj = 0;
                int dh = 0;
                countXD(dog, XD, &detH, &DXD, d, i, j, layer);
                if (abs(DXD) < 0.03)
                {
                    extrema[exIndex].at<short>(i, j) = 0;
                }
            }
        }
    }
}

void EliminateEdge(Mat *extrema, int exIndex, Mat *dog, int cols, int rows, int layer, int height)
{
    for (int j = 1; j < cols - 1; j++)
    {
        for (int i = 1; i < rows - 1; i++)
        {
            if (extrema[exIndex].at<short>(i, j) == 255)
            {
                double detHs = 0.0;
                double trHs = 0.0;
                double dxx = 0.0;
                double dxy = 0.0;
                double dyy = 0.0;
                double r = 10.0;
                double center = dog[layer].at<double>(i, j);

                dxx = dog[layer].at<double>(i, j + 1) + dog[layer].at<double>(i, j - 1) - 2.0 * center;
                dxy = (dog[layer].at<double>(i + 1, j + 1) - dog[layer].at<double>(i + 1, j - 1) - dog[layer].at<double>(i - 1, j + 1) + dog[layer].at<double>(i - 1, j - 1)) / 4.0;
                dyy = dog[layer].at<double>(i + 1, j) + dog[layer].at<double>(i - 1, j) - 2.0 * center;

                trHs = dxx + dyy;
                detHs = dxx * dyy - dxy * dxy;

                if (pow(trHs, 2) / detHs > pow((r + 1), 2) / r)
                {
                    extrema[exIndex].at<short>(i, j) = 0;
                }
            }
        }
    }
}

Mat DoG(Mat src, Mat second)
{
    Mat dog = Mat(src.rows, src.cols, dbl);
    src.convertTo(src, dbl);
    second.convertTo(second, dbl);
    for (int j = 0; j < src.cols; j++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            dog.at<double>(i, j) = abs(src.at<double>(i, j) - second.at<double>(i, j));
        }
    }
    return dog;
}

void createGaussPyramid(Mat *src, Mat *gauss)
{
    GaussianBlur(*src, gauss[0], Size(3, 3), (pow(2, 0.5) / 2) * pow(pow(2, 0.5), 0));
    for (int m = 0; m < 4; m++)
    {
        GaussianBlur(gauss[m], gauss[m + 1], Size(3, 3), (pow(2, 0.5) / 2) * pow(pow(2, 0.5), m + 1));
    }
}

void createDogPyramid(Mat *dog, Mat *gauss)
{
    for (int m = 0; m < 4; m++)
    {
        dog[m] = DoG(gauss[m], gauss[m + 1]);
    }
}

void show(Mat *toShow, int masLength, string title)
{
    for (int m = 0; m < masLength; m++)
    {
        toShow[m].convertTo(toShow[m], uchr);
        namedWindow(title + to_string(m));
        imshow(title + to_string(m), toShow[m]);
        toShow[m].convertTo(toShow[m], dbl);
    }
}

void show(Mat toShow1, Mat toShow2, string title)
{
    toShow1.convertTo(toShow1, uchr);
    toShow2.convertTo(toShow2, uchr);
    namedWindow(title);
    imshow(title, toShow1 + toShow2);
}

void FullExtrema(Mat *extremaFirst, int cols, int rows, Mat *extremaSecond, int cols1, int rows1, Mat *extremaThird, int cols2, int rows2)
{
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            if (extremaFirst[1].at<short>(i, j) == 255)
                extremaFirst[0].at<short>(i, j) = 255;
        }
    }
    for (int j = 0; j < cols1; j++)
    {
        for (int i = 0; i < rows1; i++)
        {
            if (extremaSecond[1].at<short>(i, j) == 255)
                extremaSecond[0].at<short>(i, j) = 255;
        }
    }
    for (int j = 0; j < cols2; j++)
    {
        for (int i = 0; i < rows2; i++)
        {
            if (extremaThird[1].at<short>(i, j) == 255)
                extremaThird[0].at<short>(i, j) = 255;
        }
    }
    for (int j = 0; j < cols1; j++)
    {
        for (int i = 0; i < rows1; i++)
        {
            if (extremaSecond[0].at<short>(i, j) == 255)
                extremaFirst[0].at<short>(i * 2, j * 2) = 255;
        }
    }
    for (int j = 0; j < cols2; j++)
    {
        for (int i = 0; i < rows2; i++)
        {
            if (extremaThird[0].at<short>(i, j) == 255)
                extremaFirst[0].at<short>(i * 4, j * 4) = 255;
        }
    }
}

int main(int argc, char *argv[])
{
    Mat extremaFirst[2];
    Mat extremaSecond[2];
    Mat extremaThird[2];
    Mat gaussFirst[5];
    Mat gaussSecond[5];
    Mat gaussThird[5];
    Mat dogFirst[4];
    Mat dogSecond[4];
    Mat dogThird[4];
    Mat src;
    Mat src1;
    Mat src2;
    src = imread(argv[1], uchr);
    src.convertTo(src, dbl);
    src1 = Decrease(src);
    src2 = Decrease(src1);
    int cols = src.cols;
    int rows = src.rows;
    int cols1 = src1.cols;
    int rows1 = src1.rows;
    int cols2 = src2.cols;
    int rows2 = src2.rows;
    Mat extrema = Mat(rows, cols, shrt);
    extremaFirst[0] = Mat(rows, cols, shrt);
    extremaFirst[1] = Mat(rows, cols, shrt);
    extremaSecond[0] = Mat(rows1, cols1, shrt);
    extremaSecond[1] = Mat(rows1, cols1, shrt);
    extremaThird[0] = Mat(rows2, cols2, shrt);
    extremaThird[1] = Mat(rows2, cols2, shrt);

    //First octave////////////////////////////////////////////////////////////////////////////////
    createGaussPyramid(&src, gaussFirst);
    // show(gaussFirst, 5, "Gauss1.");
    createDogPyramid(dogFirst, gaussFirst);
    // show(dogFirst, 4, "Dog1.");

    findExtrema(extremaFirst, 0, dogFirst, cols, rows, 1);
    findExtrema(extremaFirst, 1, dogFirst, cols, rows, 2);
    // show(extremaFirst[0], extremaFirst[1], "(1)Extrema");

    Clarify(extremaFirst, 0, dogFirst, cols, rows, 1, 4);
    Clarify(extremaFirst, 1, dogFirst, cols, rows, 2, 4);
    // show(extremaFirst[0], extremaFirst[1], "(1)Accurate keypoint localization");

    SmallCheck(extremaFirst, 0, dogFirst, cols, rows, 1, 4);
    SmallCheck(extremaFirst, 1, dogFirst, cols, rows, 2, 4);
    // show(extremaFirst[0], extremaFirst[1], "(1)Check keypoint for small");

    EliminateEdge(extremaFirst, 0, dogFirst, cols, rows, 1, 4);
    EliminateEdge(extremaFirst, 1, dogFirst, cols, rows, 2, 4);
    show(extremaFirst[0], extremaFirst[1], "(1)Eliminating edge responses");

    //Second octave////////////////////////////////////////////////////////////////////////////////
    createGaussPyramid(&src1, gaussSecond);
    // show(gaussSecond, 5, "Gauss2.");
    createDogPyramid(dogSecond, gaussSecond);
    // show(dogSecond, 4, "Dog2.");

    findExtrema(extremaSecond, 0, dogSecond, cols1, rows1, 1);
    findExtrema(extremaSecond, 1, dogSecond, cols1, rows1, 2);
    // show(extremaSecond[0], extremaSecond[1], "(2)Extrema");

    Clarify(extremaSecond, 0, dogSecond, cols1, rows1, 1, 4);
    Clarify(extremaSecond, 1, dogSecond, cols1, rows1, 2, 4);
    // show(extremaSecond[0], extremaSecond[1], "(2)Accurate keypoint localization");

    SmallCheck(extremaSecond, 0, dogSecond, cols1, rows1, 1, 4);
    SmallCheck(extremaSecond, 1, dogSecond, cols1, rows1, 2, 4);
    // show(extremaSecond[0], extremaSecond[1], "(2)Check keypoint for small");

    EliminateEdge(extremaSecond, 0, dogSecond, cols1, rows1, 1, 4);
    EliminateEdge(extremaSecond, 1, dogSecond, cols1, rows1, 2, 4);
    show(extremaSecond[0], extremaSecond[1], "(2)Eliminating edge responses");

    //Third octave///////////////////////////////////////////////////////////////////////////////
    createGaussPyramid(&src2, gaussThird);
    // show(gaussThird, 5, "Gauss3.");
    createDogPyramid(dogThird, gaussThird);
    // show(dogThird, 4, "Dog3.");

    findExtrema(extremaThird, 0, dogThird, cols2, rows2, 1);
    findExtrema(extremaThird, 1, dogThird, cols2, rows2, 2);
    // show(extremaThird[0], extremaThird[1], "(3)Extrema");

    Clarify(extremaThird, 0, dogThird, cols2, rows2, 1, 4);
    Clarify(extremaThird, 1, dogThird, cols2, rows2, 2, 4);
    // show(extremaThird[0], extremaThird[1], "(3)Accurate keypoint localization");

    SmallCheck(extremaThird, 0, dogThird, cols2, rows2, 1, 4);
    SmallCheck(extremaThird, 1, dogThird, cols2, rows2, 2, 4);
    // show(extremaThird[0], extremaThird[1], "(3)Check keypoint for small");

    EliminateEdge(extremaThird, 0, dogThird, cols2, rows2, 1, 4);
    EliminateEdge(extremaThird, 1, dogThird, cols2, rows2, 2, 4);
    show(extremaThird[0], extremaThird[1], "(3)Eliminating edge responses");

    FullExtrema(extremaFirst, cols, rows, extremaSecond, cols1, rows1, extremaThird, cols2, rows2);
    extremaFirst[0].convertTo(extremaFirst[0], uchr);
    namedWindow("All extrema");
    imshow("All extrema", extremaFirst[0]);

    src.convertTo(src, uchr);
    namedWindow("src");
    imshow("src", src);
    waitKey(0);
    return 0;
}
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

constexpr float fx = 528;
constexpr float fy = 528;
constexpr float cx = 320;
constexpr float cy = 240;

int pattern_x[7] = {0, 1, 2, 1, 0, -1, -2};
int pattern_y[7] = {-2, -1, 0, 1, 2, 1, 0};

typedef double NumType;
typedef Eigen::Matrix<NumType, 8, 1> Point;
typedef std::vector<Point, Eigen::aligned_allocator<Point>> PointVec;

double BilinearInterpolation(const cv::Mat &im, double x, double y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    double dx = x - u;
    double dy = y - v;
    // if (dx < 1e-3)
    //     dx = 0;
    // if (dy < 1e-3)
    //     dy = 0;

    // std::cout << "x - " << x << " y - " << y << " u - " << u << " v - " << v << " dx - " << dx << " dy - " << dy << std::endl;
    return (im.at<float>(v, u) * (1 - dx) + im.at<float>(v, u + 1) * dx) * (1 - dy) + (im.at<float>(v + 1, u) * (1 - dx) + im.at<float>(v + 1, u + 1) * dx) * dy;
}

PointVec CreatePoints(const cv::Mat &im, const cv::Mat &depth)
{
    PointVec vPoints;
    vPoints.reserve(im.cols * im.rows);
    for (int y = 2; y < im.rows - 2; ++y)
    {
        for (int x = 2; x < im.cols - 2; ++x)
        {
            NumType d = im.at<float>(y, x);
            NumType z = depth.at<float>(y, x);
            if (z > 0.3 && z < 5.0)
            {
                Point p;
                p << x, y, z, d, 0, 0, 0, 0;
                vPoints.emplace_back(std::move(p));
            }
        }
    }
    return vPoints;
}

Sophus::SE3d ComputePose(PointVec &vPoints, cv::Mat im, cv::Mat imD, const Eigen::Matrix4d &T)
{
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    Eigen::Matrix3d RKinv = T.topLeftCorner(3, 3) * K.inverse();
    Eigen::Vector3d t = T.topRightCorner(3, 1);

    for (Point &pt : vPoints)
    {
        pt(4) = -1;
        Eigen::Vector3d ptWarped = RKinv * Eigen::Vector3d(pt(0), pt(1), 1.0) * pt(2) + t;
        double u = fx * ptWarped(0) / ptWarped(2) + cx;
        double v = fy * ptWarped(1) / ptWarped(2) + cy;
        // Eigen::Vector3d ptWarped = KRKinv * Eigen::Vector3d(pt(0), pt(1), 1.0) + Kt / pt(2);
        // float u = fx * ptWarped(0) / ptWarped(2) + cx;
        // float v = fy * ptWarped(1) / ptWarped(2) + cy;
        // std::cout << ptWarped.transpose() << std::endl;

        if (u >= 1 && v >= 1 && u <= im.cols && v <= im.rows)
        {
            double rI = BilinearInterpolation(im, u, v);
            double rId = im.at<float>(v, u);
            double rD = BilinearInterpolation(imD, u, v);

            if (rI > 0 && rI < 255 && rD > 0)
            {
                pt(4) = u;
                pt(5) = v;
                pt(6) = rI;
                pt(7) = rId;
            }
        }
    }
}

int main(int argc, char **argv)
{
    cv::Mat first_image = cv::imread("/home/xyang/Downloads/playground/pics/1_rgb.png", cv::IMREAD_UNCHANGED);
    cv::Mat first_depth = cv::imread("/home/xyang/Downloads/playground/pics/1_depth.png", cv::IMREAD_UNCHANGED);
    cv::Mat second_image = cv::imread("/home/xyang/Downloads/playground/pics/2_rgb.png", cv::IMREAD_UNCHANGED);
    cv::Mat second_depth = cv::imread("/home/xyang/Downloads/playground/pics/2_depth.png", cv::IMREAD_UNCHANGED);
    first_depth.convertTo(first_depth, CV_32FC1, 1.0 / 5000);
    second_depth.convertTo(second_depth, CV_32FC1, 1.0 / 5000);

    cv::Mat first_float, second_float;
    first_image.convertTo(first_float, CV_32FC3);
    second_image.convertTo(second_float, CV_32FC3);

    cv::Mat first_gray, second_gray;
    cv::cvtColor(first_float, first_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(second_float, second_gray, cv::COLOR_BGR2GRAY);

    std::cout << "size of images: " << first_gray.size() << "\n"
              << "type of images: " << (first_gray.type() == CV_32FC1 ? "float" : "uchar") << std::endl;
    double minVal, maxVal;
    cv::minMaxIdx(first_depth, &minVal, &maxVal);
    std::cout << "first_depth: min - " << minVal << " max - " << maxVal << std::endl;
    cv::minMaxIdx(first_gray, &minVal, &maxVal);
    std::cout << "first_gray: min - " << minVal << " max - " << maxVal << std::endl;
    cv::minMaxIdx(second_depth, &minVal, &maxVal);
    std::cout << "second_depth: min - " << minVal << " max - " << maxVal << std::endl;
    cv::minMaxIdx(second_gray, &minVal, &maxVal);
    std::cout << "second_gray: min - " << minVal << " max - " << maxVal << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    PointVec vPoints = CreatePoints(first_image, first_depth);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
    std::cout << "points created: " << vPoints.size() << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    ComputePose(vPoints, second_image, second_depth, Eigen::Matrix4d::Identity());
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "time compute pose: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

    cv::Mat image_raw;
    image_raw.create(first_image.rows, first_image.cols, CV_8UC3);
    image_raw.setTo(0);
    for (auto vit = vPoints.begin(), vend = vPoints.end(); vit != vend; ++vit)
    {
        Point pt = *vit;
        // std::cout << pt.transpose() << std::endl;
        // if (pt(4) >= 0)
        image_raw.at<cv::Vec3b>(pt(1), pt(0))(1) = 254;
    }

    cv::imshow("image_raw", image_raw);
    cv::imshow("first_depth", first_depth);
    cv::imshow("first_image", first_gray);
    cv::imshow("second_image", second_gray);
    cv::imshow("second_depth", second_depth);
    cv::waitKey(0);
}
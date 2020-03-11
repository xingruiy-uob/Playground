#include <iostream>
#include "Camera.h"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "usage: ./capture_images path_to_image_folder" << std::endl;
        return -1;
    }

    std::string base_folder(argv[1]);
    OpenNI2::Camera cam;
    cv::Mat im, imDepth;
    bool bCapture = false;
    size_t currentImageId = 0;

    while (true)
    {
        if (cam.GetImages(imDepth, im))
        {
            cv::imshow("im", im);
            cv::imshow("imDepth", imDepth);
            int key = cv::waitKey(1);

            if (key == 27)
                break;

            if (key == 13)
            {
                bCapture = !bCapture;
            }

            if (bCapture)
            {
                std::stringstream ss1, ss2;
                ss1 << base_folder << currentImageId << "_depth.png";
                cv::imwrite(ss1.str(), imDepth);
                ss2 << base_folder << currentImageId << "_rgb.png";
                cv::imwrite(ss2.str(), im);

                printf("image %lu saved!\n", currentImageId);
                currentImageId++;
            }
            else
            {
                currentImageId = 0;
            }
        }
    }
}
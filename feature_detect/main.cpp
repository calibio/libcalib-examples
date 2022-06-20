#define LIBCALIB_OPENCV_INTEROP
#include "libCalib.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

void drawCross(cv::Mat &im, const cv::Point2d &pos) {

  const int shift = 5;
  const int size = 20 * (1 << shift);
  cv::Point posI((1 << shift) * pos.x, (1 << shift) * pos.y);
  cv::line(im, posI - cv::Point(size, 0), posI + cv::Point(size, 0),
           cv::Scalar(255, 0, 0), 2, cv::LINE_AA, shift);
  cv::line(im, posI - cv::Point(0, size), posI + cv::Point(0, size),
           cv::Scalar(255, 0, 0), 2, cv::LINE_AA, shift);

  return;
}

int main() {
  std::string imagesPath = std::string(SRCDIR) + "/../data/";

  //  *******************************
  //   Checkerboard detection
  //  *******************************
  {
    std::string filePath = imagesPath + "/pensylvania_c1/img_0002.png";

    cv::Mat_<uchar> im = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

    std::vector<libCalib::Point2> saddlePoints;
    const libCalib::PatternSize patternSize{6, 7};

    bool patternfound =
        libCalib::findCheckerBoard(im, patternSize, saddlePoints);

    // Subpixel refinement. Symmetry-based refinements needs to know the
    // target's world coordinates.
    std::vector<libCalib::Point3> worldPoints;
    const double squareSize = 0.10; //[m]
    for (int r = 0; r < patternSize.rows; ++r) {
      for (int c = 0; c < patternSize.columns; ++c) {
        worldPoints.emplace_back(r * squareSize, c * squareSize, 0.0);
      }
    }
    libCalib::subpixSaddlePointSymmetry(im, 10, worldPoints, saddlePoints);

    cv::Mat imColor;
    cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);
    cv::drawChessboardCorners(
        imColor, cv::Size(patternSize.columns, patternSize.rows),
        libCalib::toOpenCV<cv::Point2f>(saddlePoints), true);
    cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
    cv::imshow("IMAGE", imColor);
    cv::waitKey(0);
  }

  //*******************************
  // Partial circles grid detection
  //*******************************
  {
    std::string filePath =
        imagesPath + "/partial_circles_grid/18509006-2019-09-27-102542.png";

    cv::Mat_<uchar> im = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

    double spacing = 0.1;
    std::vector<libCalib::Point2> circleCenters;
    std::vector<double> circleRadii;
    std::vector<libCalib::Point3> worldPoints;

    bool patternfound = libCalib::findPartialCirclesGrid(
        im, libCalib::CircleContrast::BRIGHT, spacing, circleCenters,
        worldPoints, circleRadii);

    // Correction for projection of circle centers.
    libCalib::findPartialCirclesGrid(im, libCalib::CircleContrast::BRIGHT,
                                     spacing, circleCenters, worldPoints,
                                     circleRadii);

    cv::Mat imColor;
    cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < circleCenters.size(); ++i) {
      drawCross(imColor, circleCenters[i]);
    }
    cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
    cv::imshow("IMAGE", imColor);
    cv::waitKey(0);
  }

  //*******************************
  // Asymmetric circles grid
  //*******************************
  {
    std::string filePath = imagesPath + "/circle_grid/2.png";

    cv::Mat_<uchar> im = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

    double spacing = 0.1;
    std::vector<libCalib::Point2> circleCenters;
    std::vector<double> circleRadii;
    std::vector<libCalib::Point3> worldPoints;

    libCalib::PatternSize patternSize{45, 14};
    double featureSpacing = 0.018;
    const double aaSpacing = featureSpacing / std::sqrt(2.0);

    for (size_t i = 0; i < patternSize.rows; i++) {
      for (size_t j = 0; j < patternSize.columns; j++) {
        worldPoints.emplace_back(i * aaSpacing, (2 * j + i % 2) * aaSpacing,
                                 0.0);
      }
    }

    bool patternfound = libCalib::findAsymmetricCirclesGrid(
        im, patternSize, libCalib::CircleContrast::DARK, circleCenters,
        circleRadii);

    // Subpixel fit and correction for perspective bias
    libCalib::subpixCircleEllipseFit(im, worldPoints, circleCenters,
                                     circleRadii);

    cv::Mat imColor;
    cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < circleCenters.size(); ++i) {
      drawCross(imColor, circleCenters[i]);
    }
    cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
    cv::imshow("IMAGE", imColor);
    cv::waitKey(0);
  }

  return 0;
}

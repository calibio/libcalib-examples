#define LIBCALIB_OPENCV_INTEROP
#include "libCalib.h"
#include <fstream>
#include <opencv2/opencv.hpp>

// Step target. Shows how a 3d target can be used for calibration. Uses Tsai
// initialization.
// (c) Calib.io ApS

void calibrateWithLibCalib(const cv::Mat &im, const cv::Size &patternSize,
                           const std::vector<cv::Point2f> &centers) {

  std::vector<cv::Point2f> centers1;
  std::vector<cv::Point3f> Q, Q1;
  int idx = 0;
  for (int i = 0; i < patternSize.height; ++i) {
    for (int j = 0; j < patternSize.width; ++j) {
      if (i % 2 == 0) {
        Q.emplace_back(i * 4.0, j * 4.0, 0.0);
        Q1.emplace_back(i * 4.0, j * 4.0, 0.0);
        centers1.emplace_back(centers[idx]);
      } else {
        Q.emplace_back(i * 4.0, j * 4.0, -3.0);
      }

      ++idx;
    }
  }

  // do only tsai calibration
  double f, ar, cx, cy, k1;
  libCalib::Rotation Rw;
  libCalib::Translation Tw;
  libCalib::initializeCameraParametersTsai(
      libCalib::fromOpenCV(Q), libCalib::fromOpenCV(centers), im.cols, im.rows,
      f, ar, cx, cy, k1, Rw, Tw);

  std::cout << "f: " << f << '\n';
  std::cout << "ar: " << ar << '\n';
  std::cout << "cx: " << cx << '\n';
  std::cout << "cy: " << cy << '\n';
  std::cout << "k1: " << k1 << '\n';

  std::cout << "Rw: " << cv::Matx33d(Rw.matrix()) << '\n';
  std::cout << "Tw: " << Tw.toString() << std::endl;

  libCalib::Calibration cal;

  cal.addTarget(libCalib::Target(libCalib::fromOpenCV(Q)));

  libCalib::Pose pose({Rw, Tw});
  cal.addPose(pose);

  libCalib::Camera camera(libCalib::CameraModelType::OpenCV,
                          {im.cols, im.rows});
  camera.model->setVal("f", f);
  camera.model->setVal("ar", ar);
  camera.model->setVal("cx", cx);
  camera.model->setVal("cy", cy);
  camera.model->setVal("k1", 0.0);
  cal.addCamera(camera);

  std::vector<libCalib::Detection> detection = {
      libCalib::Detection(libCalib::fromOpenCV(centers))};

  cal.isInitialized = true;
  auto res = cal.optimize(detection);
  std::cout << "RPE: " << res.rpe << '\n';
  std::cout << cal.cameras[0].toString() << std::endl;
}

void calibrateWithOpenCV(const cv::Mat &im, const cv::Size &patternSize,
                         const std::vector<cv::Point2f> &centers) {

  std::vector<cv::Point2f> centers1, centers2;
  for (int i = 0; i < patternSize.height; ++i) {
    for (int j = 0; j < patternSize.width; ++j) {
      int idx = i * patternSize.width + j;
      if (i % 2 == 0) {
        centers1.push_back(centers[idx]);
      } else {
        centers2.push_back(centers[idx]);
      }
    }
  }

  std::vector<cv::Point3f> Q1, Q2;
  for (int i = 0; i < patternSize.height; ++i) {
    for (int j = 0; j < patternSize.width; ++j) {
      if (i % 2 == 0) {
        Q1.emplace_back(i * 4.0, j * 4.0, 0.0);
      } else {
        Q2.emplace_back(i * 4.0, j * 4.0, -3.0);
      }
    }
  }

  // initialize with points from only one plane
  std::vector<std::vector<cv::Point2f>> imagePoints{centers1};
  std::vector<std::vector<cv::Point3f>> objectPoints{Q1};

  cv::Matx33f K;
  cv::Vec<float, 5> k;
  std::vector<cv::Mat> rvecs, tvecs;
  cv::Mat stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;

  cv::calibrateCamera(
      objectPoints, imagePoints, im.size(), K, k, rvecs, tvecs,
      stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors,
      cv::CALIB_FIX_PRINCIPAL_POINT + cv::CALIB_FIX_K1 + cv::CALIB_FIX_K2 +
          cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST);

  // calibrate with 3d object points
  imagePoints[0].insert(imagePoints[0].end(), centers2.begin(), centers2.end());
  objectPoints[0].insert(objectPoints[0].end(), Q2.begin(), Q2.end());

  double rpe = cv::calibrateCamera(
      objectPoints, imagePoints, im.size(), K, k, rvecs, tvecs,
      stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors,
      cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_FIX_K2 + cv::CALIB_FIX_K3 +
          cv::CALIB_ZERO_TANGENT_DIST);

  std::cout << "rpe" << rpe << std::endl;
  std::cout << "K" << K << std::endl;
  std::cout << "k" << k << std::endl;
  std::cout << "stdDeviationsIntrinsics" << stdDeviationsIntrinsics
            << std::endl;
  std::cout << "stdDeviationsExtrinsics" << stdDeviationsExtrinsics
            << std::endl;

  cv::Mat R;
  cv::Rodrigues(rvecs[0], R);
  std::cout << "R: " << R << '\n';
  std::cout << "T: " << tvecs[0] << std::endl;
}

int main() {
  std::string imagesPath = std::string(SRCDIR) + "/../data/";
  std::cout << "Loading image from directory " << imagesPath << std::endl;

  cv::String filePath = imagesPath + "/step_target/full_tilted.png";

  cv::Mat_<uchar> im = cv::imread(filePath, cv::IMREAD_GRAYSCALE);

  cv::Mat imColor;
  cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);

  cv::namedWindow("IMAGE", cv::WINDOW_AUTOSIZE);
  cv::imshow("IMAGE", im);
  cv::waitKey(0);

  cv::SimpleBlobDetector::Params params;
  params.minArea = 10;
  params.maxArea = 1000;
  params.blobColor = 255;

  auto blobDetector = cv::SimpleBlobDetector::create(params);
  std::vector<cv::KeyPoint> keypoints;

  blobDetector->detect(im, keypoints);
  cv::drawKeypoints(imColor, keypoints, imColor);
  cv::imshow("IMAGE", imColor);
  cv::waitKey(0);

  cv::Size patternSize(17, 13);
  std::vector<cv::Point2f> centers;
  bool patternWasFound = cv::findCirclesGrid(
      im, patternSize, centers,
      cv::CALIB_CB_SYMMETRIC_GRID + cv::CALIB_CB_CLUSTERING, blobDetector);

  std::cout << patternWasFound << std::endl;

  std::vector<libCalib::Point2> centersLC = libCalib::fromOpenCV(centers);
  std::vector<double> radii(centers.size(), 6.0);
  libCalib::subpixCircleNonLinear(im, libCalib::CircleContrast::BRIGHT,
                                  centersLC, radii, M_PI / 4.0, 1.5);
  centers = libCalib::toOpenCV<cv::Point2f>(centersLC);

  cv::drawChessboardCorners(imColor, patternSize, centers, patternWasFound);
  cv::imshow("IMAGE", imColor);
  cv::waitKey(0);

  calibrateWithOpenCV(im, patternSize, centers);
  calibrateWithLibCalib(im, patternSize, centers);

  return 0;
}

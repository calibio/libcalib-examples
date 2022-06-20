#define LIBCALIB_OPENCV_INTEROP
#include "libCalib.h"
#include <opencv2/opencv.hpp>

// Stereo Calibration. Shows how a simple stereo-calibration setup is
// calibrated with libCalib.
// (c) Calib.io ApS

void calibrateWithLibCalib(const std::vector<cv::Mat_<uchar>> &images1,
                           const std::vector<cv::Mat_<uchar>> &images2) {
  assert(images1.size() == images2.size());

  std::cout << "Calibrating with libCalib" << std::endl;

  // construct a calibration object. this is the main interface to gather data
  // and initialize/optimize.
  libCalib::Calibration calibration;

  libCalib::PatternSize ps;
  ps.rows = 27 - 1;
  ps.columns = 39 - 1;

  const double featureSpacing = 0.025; //[m]

  // add the arget
  std::vector<libCalib::Point3> wp;
  for (size_t i = 0; i < ps.rows; ++i) {
    for (size_t j = 0; j < ps.columns; ++j) {
      wp.emplace_back(j * featureSpacing, i * featureSpacing, 0.0);
    }
  }

  calibration.addTarget(wp);

  // add the cameras
  libCalib::ImageSize imageSize = {images1[0].cols, images1[0].rows};
  libCalib::Camera camera1(libCalib::CameraModelType::OpenCV, imageSize);

  camera1.model->setState("f", libCalib::ParameterState::FREE);
  camera1.model->setState("ar", libCalib::ParameterState::FREE);
  camera1.model->setState("cx", libCalib::ParameterState::FREE);
  camera1.model->setState("cy", libCalib::ParameterState::FREE);
  camera1.model->setState("k1", libCalib::ParameterState::FREE);
  camera1.model->setState("k2", libCalib::ParameterState::FREE);
  camera1.model->setState("k3", libCalib::ParameterState::FIXED);
  camera1.model->setState("p1", libCalib::ParameterState::FIXED);
  camera1.model->setState("p2", libCalib::ParameterState::FIXED);

  // camera 2 has the same model and parameters
  libCalib::Camera camera2(camera1);

  calibration.addCamera(camera1);
  calibration.addCamera(camera2);

  // simple vector of feature points
  std::vector<libCalib::Point2> featurePointLocations;

  size_t nImages = images1.size();
  std::vector<libCalib::Detection> detections;
  for (size_t i = 0; i < nImages; ++i) {
    std::cout << "Detecting features " << i + 1 << "/" << nImages << std::endl;

    std::vector<libCalib::Point2> imagePoints1;
    std::vector<size_t> featureIds1;
    bool s1 =
        libCalib::findChAruCoBoard(images1[i], ps, imagePoints1, featureIds1,
                                   libCalib::ArucoDictionary::DICT_4X4);
    if (s1) {
      const int kernelHalfWidth = 5;
      libCalib::subpixSaddlePointPolynomium(images1[i], kernelHalfWidth,
                                            imagePoints1);
    }

    std::vector<libCalib::Point2> imagePoints2;
    std::vector<size_t> featureIds2;
    bool s2 =
        libCalib::findChAruCoBoard(images2[i], ps, imagePoints2, featureIds2,
                                   libCalib::ArucoDictionary::DICT_4X4);
    if (s2) {
      const int kernelHalfWidth = 5;
      libCalib::subpixSaddlePointPolynomium(images2[i], kernelHalfWidth,
                                            imagePoints2);
    }

    if (s1 || s2) {
      size_t poseId = calibration.addPose();

      if (s1) {
        detections.push_back({imagePoints1, featureIds1, 0, poseId, 0});
      }
      if (s2) {
        detections.push_back({imagePoints2, featureIds2, 1, poseId, 0});
      }
    } else {
      std::cout << "Could not detect features on images " << i << std::endl;
    }
  }

  libCalib::InitializationResult initResult;
  initResult = calibration.initialize(detections);
  std::cout << initResult.statusString << '\n';

  libCalib::OptimizationSettings optSettings;
  libCalib::OptimizationResult optResult;
  optResult = calibration.optimize(detections, optSettings);

  std::cout << optResult.statusString << '\n';
  std::cout << "RPE: " << optResult.rpe << '\n';

  std::cout << "Camera 0:\n" << calibration.cameras[0].toString() << '\n';
  std::cout << "Camera 1:\n" << calibration.cameras[1].toString() << '\n';

  for (const auto &p : calibration.poses) {
    std::cout << p.toString() << '\n';
  }
  std::cout << std::flush;
}

int main() {
  std::string imagesPath = std::string(SRCDIR) + "/../data/charuco_stereo";
  std::cout << "Loading images from directory " << imagesPath << std::endl;

  std::vector<cv::String> filePaths1;
  cv::glob(imagesPath + "/cam0/*.png", filePaths1, false);
  std::vector<cv::Mat_<uchar>> images1;
  for (const auto &p : filePaths1) {
    images1.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));
  }

  std::vector<cv::String> filePaths2;
  cv::glob(imagesPath + "/cam1/*.png", filePaths2, false);
  std::vector<cv::Mat_<uchar>> images2;
  for (const auto &p : filePaths2) {
    images2.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));
  }

  calibrateWithLibCalib(images1, images2);

  return 0;
}

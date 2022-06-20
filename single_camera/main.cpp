#define LIBCALIB_OPENCV_INTEROP
#include "libCalib.h"
#include <opencv2/opencv.hpp>

// Single Camera. Demonstrates how a single camera can be calibrated and
// compares with OpenCV methods.
// (c) Calib.io ApS

void calibrateWithOpenCV(const std::vector<cv::String> &filePaths) {
  std::cout << "Calibrating with OpenCV" << std::endl;
  int64_t tStart = cv::getTickCount();

  const cv::Size patternSize(7 - 1, 8 - 1);
  const double featureSpacing = 0.050; //[m]

  std::vector<std::vector<cv::Point2f>> corners;
  cv::Size imageSize;
  for (auto &fp : filePaths) {

    cv::Mat im = cv::imread(fp, cv::IMREAD_GRAYSCALE);

    std::vector<cv::Point2f> cornersI;
    cv::findChessboardCorners(im, patternSize, cornersI);

    if (cornersI.size() == patternSize.area()) {
      cv::cornerSubPix(im, cornersI, cv::Size(5, 5), cv::Size(1, 1),
                       cv::TermCriteria());

      corners.push_back(cornersI);
    }

    imageSize = im.size();
  }

  // world points
  std::vector<cv::Point3f> wpI;
  for (int i = 0; i < patternSize.height; ++i) {
    for (int j = 0; j < patternSize.width; ++j) {
      wpI.emplace_back(i * featureSpacing, j * featureSpacing, 0.0);
    }
  }

  const auto nImages = corners.size();

  std::vector<std::vector<cv::Point3f>> wp;
  std::fill_n(std::back_inserter(wp), nImages, wpI);

  int flags = cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST;
  cv::Mat K, k;
  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> stdDevIntrinsics, stdDevExtrinsics, perViewErrors;

  double error = cv::calibrateCamera(wp, corners, imageSize, K, k, rvecs, tvecs,
                                     stdDevIntrinsics, stdDevExtrinsics,
                                     perViewErrors, flags);

  int64_t tStop = cv::getTickCount();
  double elapsedTime =
      static_cast<double>(tStop - tStart) / cv::getTickFrequency();

  std::cout << "Reprojection error: " << error << std::endl;
  std::cout << "K: " << K << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "std.dev of fx " << stdDevIntrinsics[0] << '\n';
  std::cout << "std.dev of fy " << stdDevIntrinsics[1] << '\n';
  std::cout << "std.dev of cx " << stdDevIntrinsics[2] << '\n';
  std::cout << "std.dev of cy " << stdDevIntrinsics[3] << '\n';
  std::cout << "std.dev of k1 " << stdDevIntrinsics[4] << '\n';
  std::cout << "std.dev of k2 " << stdDevIntrinsics[5] << '\n';

  std::cout << "time: " << elapsedTime << " s" << std::endl;
}

void calibrateWithLibCalib(const std::vector<cv::String> &filePaths) {

  std::cout << "Calibrating with libCalib" << std::endl;
  int64_t tStart = cv::getTickCount();

  // construct a calibration object. this is the main interface to gather data
  // and initialize/optimize.
  libCalib::Calibration calibration;

  libCalib::PatternSize dp;
  dp.rows = 7 - 1;
  dp.columns = 8 - 1;

  const double featureSpacing = 0.050; //[m]

  // add target
  std::vector<libCalib::Point3> wp;
  for (size_t i = 0; i < dp.rows; ++i) {
    for (size_t j = 0; j < dp.columns; ++j) {
      wp.emplace_back(j * featureSpacing, i * featureSpacing, 0.0);
    }
  }

  calibration.addTarget(wp);

  libCalib::ImageSize imageSize;

  // simple vector of feature points
  std::vector<libCalib::Point2> featurePointLocations;
  size_t nPoses = 0;
  std::vector<libCalib::Detection> detections;
  for (auto &fp : filePaths) {

    cv::Mat_<uchar> im = cv::imread(fp, cv::IMREAD_GRAYSCALE);
    imageSize = {im.cols, im.rows};

    std::vector<libCalib::Point2> featurePointsI;

    libCalib::findCheckerBoard(im, dp, featurePointsI);

    if (featurePointsI.size() == static_cast<size_t>(dp.rows * dp.columns)) {

      const int kernelHalfWidth = 5;
      libCalib::subpixSaddlePointPolynomium(im, kernelHalfWidth,
                                            featurePointsI);

      if (featurePointsI.size() == static_cast<size_t>(dp.rows * dp.columns)) {
        calibration.addPose();

        libCalib::Detection detection(featurePointsI, 0, nPoses++, 0);
        detections.push_back(detection);

        // copy into vector for visualization
        featurePointLocations.insert(featurePointLocations.end(),
                                     featurePointsI.begin(),
                                     featurePointsI.end());
      }
    }
  }

  // add a camera
  libCalib::Camera camera(libCalib::CameraModelType::OpenCV, imageSize);

  camera.model->setState("f", libCalib::ParameterState::FREE);
  camera.model->setState("ar", libCalib::ParameterState::FREE);
  camera.model->setState("cx", libCalib::ParameterState::FREE);
  camera.model->setState("cy", libCalib::ParameterState::FREE);
  camera.model->setState("k1", libCalib::ParameterState::FREE);
  camera.model->setState("k2", libCalib::ParameterState::FREE);
  camera.model->setState("k3", libCalib::ParameterState::FIXED);
  camera.model->setState("p1", libCalib::ParameterState::FIXED);
  camera.model->setState("p2", libCalib::ParameterState::FIXED);

  calibration.addCamera(camera);

  //  calibration.saveToDisk("single_camera_detected.json");

  //  calibration.loadFromDisk("single_camera_detected.json");
  //  libCalib::Camera &camera = calibration.getCameras()[0];

  calibration.initialize(detections);

  libCalib::OptimizationSettings optSettings;
  auto result = calibration.optimize(detections, optSettings);

  auto cov = calibration.estimateCovariance(detections);

  int64_t tStop = cv::getTickCount();
  double elapsedTime =
      static_cast<double>(tStop - tStart) / cv::getTickFrequency();

  std::cout << "Reprojection error: " << result.rpe << std::endl;

  std::cout << calibration.cameras[0].toString() << std::endl;

  for (size_t i = 0; i < cov.labels.size(); ++i) {
    std::cout << "std.dev of " << cov.labels[i].second << ' '
              << std::sqrt(cov.covarianceMatrix(i, i)) << std::endl;
  }

  std::cout << "time: " << elapsedTime << " s" << std::endl;

  // copy all residuals into simple vector
  std::vector<libCalib::Point2> residuals;
  for (const auto &dr : result.detectionResiduals) {
    for (const auto &r : dr.residuals) {
      residuals.push_back(r.error);
    }
  }

  cv::Mat residualDiagram =
      libCalib::errorDirectionsMap(featurePointLocations, residuals, imageSize);
  cv::imwrite("errorDirections.png", residualDiagram);

  int64_t tStartKDE = cv::getTickCount();

  std::pair<cv::Mat, double> density =
      libCalib::kernelDensityEstimate(result.detectionResiduals);
  std::cout << "KDE: "
            << static_cast<double>(cv::getTickCount() - tStartKDE) /
                   cv::getTickFrequency()
            << 's' << std::endl;

  cv::normalize(density.first, density.first, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imwrite("density.png", density.first);

  result.saveToDisk("result.json");
}

int main() {
  std::string imagesPath = std::string(SRCDIR) + "/../data/pensylvania_c1";
  std::cout << "Loading images from directory " << imagesPath << std::endl;

  std::vector<cv::String> filePaths;
  cv::glob(imagesPath + "/*.png", filePaths, false);
  //  filePaths.resize(5);

  calibrateWithOpenCV(filePaths);
  calibrateWithLibCalib(filePaths);

  return 0;
}

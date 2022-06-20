#define LIBCALIB_OPENCV_INTEROP
#include "libCalib.h"

#include <opencv2/opencv.hpp>

// OneShot Factory Calibration with a OneShot 3d target
// (c) Calib.io ApS
int main() {

  std::string imageFilePath =
      std::string(SRCDIR) + "/../data/one_shot/img0.png";
  cv::Mat_<uchar> image = cv::imread(imageFilePath, cv::IMREAD_GRAYSCALE);

  std::cout << "Calibrating with libCalib" << std::endl;
  int64_t tStart = cv::getTickCount();

  // construct a calibration object
  libCalib::Calibration calibration;

  libCalib::Target target;
  std::vector<size_t> markerIds;
  libCalib::createOneShotPyramidTarget(
      33, 21, 0.025, {0.0, 2 * 0.0254, 4 * 0.0254, 6 * 0.0254, 8 * 0.0254},
      target, markerIds);
  calibration.addTarget(target);

  calibration.addPose();

  // add a camera
  libCalib::ImageSize imageSize(image.cols, image.rows);

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

  std::vector<libCalib::Point2> featurePoints;
  std::vector<size_t> featureIds;

  libCalib::find3DCirclesTarget(image, target.objectPoints, markerIds, 3430,
                                featurePoints, featureIds);

  std::vector<libCalib::Detection> detections;
  detections.emplace_back(featurePoints, featureIds);

  libCalib::InitializationSettings initSettings;
  initSettings.method = libCalib::InitializationMethod::TSAI;

  calibration.initialize(detections, initSettings);

  libCalib::OptimizationSettings optSettings;
  auto result = calibration.optimize(detections, optSettings);

  auto cov = calibration.estimateCovariance(detections);

  int64_t tStop = cv::getTickCount();
  double elapsedTime =
      static_cast<double>(tStop - tStart) / cv::getTickFrequency();

  std::cout << "Reprojection error: " << result.rpe << "px" << std::endl;

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

  cv::Mat errorMagnitudes =
      libCalib::errorMagnitudesMap(featurePoints, residuals, imageSize);
  cv::imwrite("errorMagnitudes.png", errorMagnitudes);

  cv::Mat errorDirections =
      libCalib::errorDirectionsMap(featurePoints, residuals, imageSize);
  cv::imwrite("errorDirections.png", errorDirections);

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

  return 0;
}

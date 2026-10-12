#ifndef WAFER_ALIGNMENT_HPP
#define WAFER_ALIGNMENT_HPP

#include <opencv2/opencv.hpp>
#include "WaferData.hpp"

namespace WaferVision {

enum class AlignmentMethod {
    CENTER_OF_MASS,
    HOUGH_CIRCLE,
    TEMPLATE_MATCHING,
    EDGE_BASED,
    HYBRID
};

struct AlignmentResult {
    cv::Point2f center;
    double rotation;
    double confidence;
    cv::Mat transformationMatrix;
    bool success;
};

class WaferAlignment {
public:
    WaferAlignment();
    ~WaferAlignment();

    // Alignment methods
    AlignmentResult alignWafer(const cv::Mat& image, AlignmentMethod method = AlignmentMethod::HYBRID);
    AlignmentResult alignByCenterOfMass(const cv::Mat& image);
    AlignmentResult alignByHoughCircle(const cv::Mat& image);
    AlignmentResult alignByTemplateMatching(const cv::Mat& image, const cv::Mat& templateImage);
    AlignmentResult alignByEdgeDetection(const cv::Mat& image);
    AlignmentResult alignHybrid(const cv::Mat& image);

    // Apply alignment
    cv::Mat applyAlignment(const cv::Mat& image, const AlignmentResult& result);

    // Flat detection (for wafer flat/notch)
    cv::Point2f detectFlat(const cv::Mat& image);
    double calculateFlatAngle(const cv::Mat& image, const cv::Point2f& center);

    // Notch detection
    cv::Point2f detectNotch(const cv::Mat& image);
    double calculateNotchAngle(const cv::Mat& image, const cv::Point2f& center);

    // Configuration
    void setExpectedDiameter(double diameter);
    void setTolerance(double tolerance);
    void setAlignmentMethod(AlignmentMethod method);

private:
    double m_expectedDiameter;
    double m_tolerance;
    AlignmentMethod m_method;

    // Helper functions
    cv::Mat preprocessForAlignment(const cv::Mat& image);
    cv::Point2f findCenterOfMass(const cv::Mat& binaryImage);
    std::vector<cv::Vec3f> detectCircles(const cv::Mat& image);
    cv::Mat createWaferTemplate(int diameter);
};

} // namespace WaferVision

#endif // WAFER_ALIGNMENT_HPP

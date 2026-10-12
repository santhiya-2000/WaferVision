#ifndef METROLOGY_HPP
#define METROLOGY_HPP

#include <opencv2/opencv.hpp>
#include "WaferData.hpp"

namespace WaferVision {

enum class MeasurementType {
    CRITICAL_DIMENSION,
    LINE_WIDTH,
    SPACE_WIDTH,
    OVERLAY_ERROR,
    FOCUS_METRIC,
    THICKNESS,
    ROUGHNESS
};

struct MeasurementResult {
    MeasurementType type;
    double value;
    double uncertainty;
    cv::Point2f location;
    std::string unit;
    bool valid;
};

class Metrology {
public:
    Metrology();
    ~Metrology();

    // Main metrology workflow
    MetrologyData runMetrology(const cv::Mat& image, const WaferMap& waferMap);
    
    // Critical dimension measurement
    MeasurementResult measureCriticalDimension(const cv::Mat& image, const cv::Point2f& location);
    std::vector<MeasurementResult> measureCriticalDimensions(const cv::Mat& image, 
                                                             const std::vector<cv::Point2f>& locations);
    
    // Line/space width measurement
    MeasurementResult measureLineWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end);
    MeasurementResult measureSpaceWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end);
    
    // Overlay error measurement
    MeasurementResult measureOverlayError(const cv::Mat& currentLayer, const cv::Mat& referenceLayer);
    
    // Focus metrics
    MeasurementResult calculateFocusMetric(const cv::Mat& image);
    double calculateSharpnessMetric(const cv::Mat& image);
    double calculateContrastMetric(const cv::Mat& image);
    
    // Thickness estimation (interferometry-based)
    MeasurementResult estimateThickness(const cv::Mat& interferogram);
    
    // Surface roughness
    MeasurementResult measureRoughness(const cv::Mat& surfaceImage);
    
    // Edge detection for measurements
    std::vector<cv::Point2f> detectEdges(const cv::Mat& image, const cv::Rect& roi);
    cv::Point2f findEdgePosition(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& direction);
    
    // Sub-pixel refinement
    cv::Point2f refineEdgeSubpixel(const cv::Mat& image, const cv::Point2f& edge);
    
    // Configuration
    void setCalibrationFactor(double factor);
    void setMeasurementTolerance(double tolerance);

private:
    double m_calibrationFactor;
    double m_measurementTolerance;
    
    // Helper functions
    cv::Mat preprocessForMetrology(const cv::Mat& image);
    double calculateLineProfile(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end);
    std::vector<double> extractLineProfile(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end);
    double fitGaussian(const std::vector<double>& profile);
};

} // namespace WaferVision

#endif // METROLOGY_HPP

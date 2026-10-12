#ifndef EDGE_REDUCTION_HPP
#define EDGE_REDUCTION_HPP

#include <opencv2/opencv.hpp>

namespace WaferVision {

class EdgeReduction {
public:
    EdgeReduction();
    ~EdgeReduction();

    // Main edge reduction methods
    cv::Mat reduceEdgeArtifacts(const cv::Mat& image);
    cv::Mat suppressWaferEdges(const cv::Mat& image, const cv::Point2f& center, double radius);
    cv::Mat adaptiveEdgeFilter(const cv::Mat& image);

    // Edge detection and masking
    cv::Mat detectWaferEdges(const cv::Mat& image);
    cv::Mat createEdgeMask(const cv::Mat& image, const cv::Point2f& center, double radius, double margin);
    
    // Gradient-based filtering
    cv::Mat gradientBasedFilter(const cv::Mat& image);
    cv::Mat computeGradientMagnitude(const cv::Mat& image);
    
    // Morphological operations for edge reduction
    cv::Mat morphologicalEdgeReduction(const cv::Mat& image);
    
    // Configuration
    void setEdgeMargin(double margin);
    void setEdgeThreshold(double threshold);

private:
    double m_edgeMargin;
    double m_edgeThreshold;
};

} // namespace WaferVision

#endif // EDGE_REDUCTION_HPP

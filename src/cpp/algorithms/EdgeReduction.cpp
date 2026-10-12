#include "EdgeReduction.hpp"
#include <opencv2/imgproc.hpp>

namespace WaferVision {

EdgeReduction::EdgeReduction() 
    : m_edgeMargin(20.0), m_edgeThreshold(50.0) {
}

EdgeReduction::~EdgeReduction() = default;

cv::Mat EdgeReduction::reduceEdgeArtifacts(const cv::Mat& image) {
    cv::Mat edges = detectWaferEdges(image);
    cv::Mat mask = createEdgeMask(image, cv::Point2f(image.cols/2.0f, image.rows/2.0f), 
                                  std::min(image.cols, image.rows)/2.0, m_edgeMargin);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

cv::Mat EdgeReduction::suppressWaferEdges(const cv::Mat& image, const cv::Point2f& center, double radius) {
    cv::Mat mask = createEdgeMask(image, center, radius, m_edgeMargin);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

cv::Mat EdgeReduction::adaptiveEdgeFilter(const cv::Mat& image) {
    cv::Mat gradient = computeGradientMagnitude(image);
    
    cv::Mat mask;
    cv::threshold(gradient, mask, m_edgeThreshold, 255, cv::THRESH_BINARY);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

cv::Mat EdgeReduction::detectWaferEdges(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat edges;
    cv::Canny(gray, edges, m_edgeThreshold, m_edgeThreshold * 2);
    
    return edges;
}

cv::Mat EdgeReduction::createEdgeMask(const cv::Mat& image, const cv::Point2f& center, double radius, double margin) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    
    // Create circular mask for wafer edge
    cv::circle(mask, center, static_cast<int>(radius + margin), cv::Scalar(255), -1);
    cv::circle(mask, center, static_cast<int>(radius - margin), cv::Scalar(0), -1);
    
    // Add flat/notch region if needed
    cv::Rect flatRegion(center.x - radius - margin, center.y - margin/2, margin, margin);
    cv::rectangle(mask, flatRegion, cv::Scalar(255), -1);
    
    return mask;
}

cv::Mat EdgeReduction::gradientBasedFilter(const cv::Mat& image) {
    cv::Mat gradient = computeGradientMagnitude(image);
    
    cv::Mat mask;
    cv::threshold(gradient, mask, m_edgeThreshold, 255, cv::THRESH_BINARY);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

cv::Mat EdgeReduction::computeGradientMagnitude(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_64F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    
    cv::Mat result;
    magnitude.convertTo(result, CV_8U);
    
    return result;
}

cv::Mat EdgeReduction::morphologicalEdgeReduction(const cv::Mat& image) {
    cv::Mat edges = detectWaferEdges(image);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    cv::Mat dilated;
    cv::dilate(edges, dilated, kernel);
    
    cv::Mat mask;
    cv::threshold(dilated, mask, 127, 255, cv::THRESH_BINARY);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

void EdgeReduction::setEdgeMargin(double margin) {
    m_edgeMargin = margin;
}

void EdgeReduction::setEdgeThreshold(double threshold) {
    m_edgeThreshold = threshold;
}

} // namespace WaferVision

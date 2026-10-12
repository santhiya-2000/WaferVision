#ifndef PATTERN_PROCESSING_HPP
#define PATTERN_PROCESSING_HPP

#include <opencv2/opencv.hpp>

namespace WaferVision {

class PatternProcessing {
public:
    PatternProcessing();
    ~PatternProcessing();

    // Pattern analysis
    cv::Mat analyzePattern(const cv::Mat& image);
    cv::Mat detectPeriodicPattern(const cv::Mat& image);
    cv::Mat estimatePatternFrequency(const cv::Mat& image);
    
    // Pattern filtering
    cv::Mat filterPatternNoise(const cv::Mat& image);
    cv::Mat removePeriodicArtifacts(const cv::Mat& image);
    cv::Mat enhancePatternFeatures(const cv::Mat& image);
    
    // Pattern matching
    cv::Mat matchPattern(const cv::Mat& image, const cv::Mat& pattern);
    std::vector<cv::Point> findPatternLocations(const cv::Mat& image, const cv::Mat& pattern, double threshold);
    
    // Fourier-based pattern analysis
    cv::Mat computeFFT(const cv::Mat& image);
    cv::Mat filterFFT(const cv::Mat& fftImage, const cv::Mat& mask);
    cv::Mat computeInverseFFT(const cv::Mat& fftImage);
    
    // Configuration
    void setPatternSize(int size);
    void setMatchThreshold(double threshold);

private:
    int m_patternSize;
    double m_matchThreshold;
};

} // namespace WaferVision

#endif // PATTERN_PROCESSING_HPP

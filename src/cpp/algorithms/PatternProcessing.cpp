#include "PatternProcessing.hpp"
#include <opencv2/imgproc.hpp>
#include <complex>

namespace WaferVision {

PatternProcessing::PatternProcessing() 
    : m_patternSize(32), m_matchThreshold(0.8) {
}

PatternProcessing::~PatternProcessing() = default;

cv::Mat PatternProcessing::analyzePattern(const cv::Mat& image) {
    cv::Mat pattern = detectPeriodicPattern(image);
    cv::Mat filtered = filterPatternNoise(pattern);
    return filtered;
}

cv::Mat PatternProcessing::detectPeriodicPattern(const cv::Mat& image) {
    cv::Mat fft = computeFFT(image);
    
    // Create a mask to keep only the periodic components
    cv::Mat mask = cv::Mat::zeros(fft.size(), CV_8U);
    cv::circle(mask, cv::Point(fft.cols/2, fft.rows/2), 30, cv::Scalar(255), -1);
    
    cv::Mat filteredFFT = filterFFT(fft, mask);
    cv::Mat result = computeInverseFFT(filteredFFT);
    
    return result;
}

cv::Mat PatternProcessing::estimatePatternFrequency(const cv::Mat& image) {
    cv::Mat fft = computeFFT(image);
    
    // Shift FFT to center
    cv::Mat shifted;
    int cx = fft.cols / 2;
    int cy = fft.rows / 2;
    
    cv::Mat q0(fft, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(fft, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(fft, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(fft, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    return fft;
}

cv::Mat PatternProcessing::filterPatternNoise(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    
    cv::Mat result;
    cv::medianBlur(blurred, result, 3);
    
    return result;
}

cv::Mat PatternProcessing::removePeriodicArtifacts(const cv::Mat& image) {
    cv::Mat fft = computeFFT(image);
    
    // Remove high-frequency periodic components
    cv::Mat mask = cv::Mat::ones(fft.size(), CV_8U) * 255;
    cv::circle(mask, cv::Point(fft.cols/2, fft.rows/2), 50, cv::Scalar(0), -1);
    
    cv::Mat filteredFFT = filterFFT(fft, mask);
    cv::Mat result = computeInverseFFT(filteredFFT);
    
    return result;
}

cv::Mat PatternProcessing::enhancePatternFeatures(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat enhanced;
    cv::equalizeHist(gray, enhanced);
    
    cv::Mat sharpened;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::filter2D(enhanced, sharpened, -1, kernel);
    
    return sharpened;
}

cv::Mat PatternProcessing::matchPattern(const cv::Mat& image, const cv::Mat& pattern) {
    cv::Mat result;
    cv::matchTemplate(image, pattern, result, cv::TM_CCOEFF_NORMED);
    return result;
}

std::vector<cv::Point> PatternProcessing::findPatternLocations(const cv::Mat& image, const cv::Mat& pattern, double threshold) {
    cv::Mat result = matchPattern(image, pattern);
    
    std::vector<cv::Point> locations;
    
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (result.at<float>(y, x) > threshold) {
                locations.push_back(cv::Point(x, y));
            }
        }
    }
    
    return locations;
}

cv::Mat PatternProcessing::computeFFT(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Expand to optimal size
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, 
                      cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Convert to float
    cv::Mat floatImage;
    padded.convertTo(floatImage, CV_32F);
    
    // Compute DFT
    cv::Mat planes[] = {floatImage, cv::Mat::zeros(floatImage.size(), CV_32F)};
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);
    cv::dft(complexImage, complexImage);
    
    return complexImage;
}

cv::Mat PatternProcessing::filterFFT(const cv::Mat& fftImage, const cv::Mat& mask) {
    cv::Mat filtered;
    fftImage.copyTo(filtered);
    
    for (int y = 0; y < fftImage.rows; ++y) {
        for (int x = 0; x < fftImage.cols; ++x) {
            if (mask.at<unsigned char>(y, x) == 0) {
                filtered.at<cv::Vec2f>(y, x) = cv::Vec2f(0, 0);
            }
        }
    }
    
    return filtered;
}

cv::Mat PatternProcessing::computeInverseFFT(const cv::Mat& fftImage) {
    cv::Mat inverse;
    cv::idft(fftImage, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    cv::Mat result;
    inverse.convertTo(result, CV_8U);
    
    return result;
}

void PatternProcessing::setPatternSize(int size) {
    m_patternSize = size;
}

void PatternProcessing::setMatchThreshold(double threshold) {
    m_matchThreshold = threshold;
}

} // namespace WaferVision

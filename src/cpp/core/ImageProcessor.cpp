#include "ImageProcessor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

namespace WaferVision {

ImageProcessor::ImageProcessor() 
    : m_useGpu(false), m_gpuAvailable(false) {
    // Check for CUDA availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    m_gpuAvailable = (error == cudaSuccess && deviceCount > 0);
    
    if (m_gpuAvailable) {
        std::cout << "GPU acceleration available (" << deviceCount << " device(s))" << std::endl;
    }
}

ImageProcessor::~ImageProcessor() = default;

cv::Mat ImageProcessor::loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return image;
}

cv::Mat ImageProcessor::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Denoise
    processed = denoiseImage(image);
    
    // Normalize
    processed = normalizeImage(processed);
    
    // Enhance contrast
    processed = enhanceContrast(processed);
    
    return processed;
}

cv::Mat ImageProcessor::normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    return normalized;
}

cv::Mat ImageProcessor::denoiseImage(const cv::Mat& image) {
    if (m_useGpu && m_gpuAvailable) {
        return gpuDenoise(image);
    }
    
    cv::Mat denoised;
    cv::fastNlMeansDenoising(image, denoised, 10, 7, 21);
    return denoised;
}

cv::Mat ImageProcessor::enhanceContrast(const cv::Mat& image) {
    if (m_useGpu && m_gpuAvailable) {
        return gpuEnhanceContrast(image);
    }
    
    cv::Mat enhanced;
    cv::equalizeHist(image, enhanced);
    return enhanced;
}

cv::Mat ImageProcessor::sharpenImage(const cv::Mat& image) {
    cv::Mat sharpened;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::filter2D(image, sharpened, -1, kernel);
    return sharpened;
}

cv::Mat ImageProcessor::adjustBrightness(const cv::Mat& image, double alpha, double beta) {
    cv::Mat adjusted;
    image.convertTo(adjusted, -1, alpha, beta);
    return adjusted;
}

cv::Mat ImageProcessor::rotateImage(const cv::Mat& image, double angle) {
    cv::Mat rotated;
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(image, rotated, rotationMatrix, image.size());
    return rotated;
}

cv::Mat ImageProcessor::scaleImage(const cv::Mat& image, double scale) {
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return scaled;
}

cv::Mat ImageProcessor::translateImage(const cv::Mat& image, int dx, int dy) {
    cv::Mat translated;
    cv::Mat translationMatrix = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(image, translated, translationMatrix, image.size());
    return translated;
}

bool ImageProcessor::isGpuAvailable() const {
    return m_gpuAvailable;
}

void ImageProcessor::setUseGpu(bool useGpu) {
    m_useGpu = useGpu && m_gpuAvailable;
}

cv::Mat ImageProcessor::gpuDenoise(const cv::Mat& image) {
    // Placeholder for GPU implementation
    // In production, this would use CUDA kernels
    cv::Mat denoised;
    cv::fastNlMeansDenoising(image, denoised, 10, 7, 21);
    return denoised;
}

cv::Mat ImageProcessor::gpuEnhanceContrast(const cv::Mat& image) {
    // Placeholder for GPU implementation
    cv::Mat enhanced;
    cv::equalizeHist(image, enhanced);
    return enhanced;
}

} // namespace WaferVision

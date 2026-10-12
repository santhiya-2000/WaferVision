#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace WaferVision {

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Image loading and preprocessing
    cv::Mat loadImage(const std::string& path);
    cv::Mat preprocessImage(const cv::Mat& image);
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat denoiseImage(const cv::Mat& image);

    // Image enhancement
    cv::Mat enhanceContrast(const cv::Mat& image);
    cv::Mat sharpenImage(const cv::Mat& image);
    cv::Mat adjustBrightness(const cv::Mat& image, double alpha, double beta);

    // Image transformation
    cv::Mat rotateImage(const cv::Mat& image, double angle);
    cv::Mat scaleImage(const cv::Mat& image, double scale);
    cv::Mat translateImage(const cv::Mat& image, int dx, int dy);

    // GPU acceleration check
    bool isGpuAvailable() const;
    void setUseGpu(bool useGpu);

private:
    bool m_useGpu;
    bool m_gpuAvailable;

    // GPU processing functions (implemented in CUDA)
    cv::Mat gpuDenoise(const cv::Mat& image);
    cv::Mat gpuEnhanceContrast(const cv::Mat& image);
};

} // namespace WaferVision

#endif // IMAGE_PROCESSOR_HPP

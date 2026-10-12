#include "Metrology.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace WaferVision {

Metrology::Metrology() 
    : m_calibrationFactor(1.0), m_measurementTolerance(0.01) {
}

Metrology::~Metrology() = default;

MetrologyData Metrology::runMetrology(const cv::Mat& image, const WaferMap& waferMap) {
    MetrologyData data;
    
    cv::Mat processed = preprocessForMetrology(image);
    
    // Calculate focus metric
    auto focusResult = calculateFocusMetric(processed);
    data.focusMetric = focusResult.value;
    
    // Measure critical dimensions at multiple locations
    std::vector<cv::Point2f> measurementPoints;
    int gridSize = 5;
    float stepX = image.cols / static_cast<float>(gridSize + 1);
    float stepY = image.rows / static_cast<float>(gridSize + 1);
    
    for (int i = 1; i <= gridSize; ++i) {
        for (int j = 1; j <= gridSize; ++j) {
            cv::Point2f point(j * stepX, i * stepY);
            measurementPoints.push_back(point);
        }
    }
    
    data.measurementPoints = measurementPoints;
    
    // Sample measurements
    if (!measurementPoints.empty()) {
        auto cdResult = measureCriticalDimension(processed, measurementPoints[0]);
        data.criticalDimension = cdResult.value;
        
        if (measurementPoints.size() > 1) {
            auto lwResult = measureLineWidth(processed, measurementPoints[0], measurementPoints[1]);
            data.lineWidth = lwResult.value;
            
            auto swResult = measureSpaceWidth(processed, measurementPoints[1], measurementPoints[2]);
            data.spaceWidth = swResult.value;
        }
    }
    
    return data;
}

MeasurementResult Metrology::measureCriticalDimension(const cv::Mat& image, const cv::Point2f& location) {
    MeasurementResult result;
    result.type = MeasurementType::CRITICAL_DIMENSION;
    result.location = location;
    result.unit = "nm";
    
    cv::Mat processed = preprocessForMetrology(image);
    
    // Extract line profile in horizontal direction
    cv::Point2f start(location.x - 50, location.y);
    cv::Point2f end(location.x + 50, location.y);
    
    std::vector<double> profile = extractLineProfile(processed, start, end);
    
    // Find edges in profile
    double threshold = cv::mean(processed).val[0] * 0.5;
    int edge1 = -1, edge2 = -1;
    
    for (size_t i = 1; i < profile.size(); ++i) {
        if (edge1 == -1 && profile[i] > threshold && profile[i-1] <= threshold) {
            edge1 = static_cast<int>(i);
        }
        if (edge2 == -1 && profile[i] < threshold && profile[i-1] >= threshold) {
            edge2 = static_cast<int>(i);
        }
    }
    
    if (edge1 != -1 && edge2 != -1) {
        double pixelWidth = edge2 - edge1;
        result.value = pixelWidth * m_calibrationFactor;
        result.uncertainty = m_measurementTolerance * result.value;
        result.valid = true;
    } else {
        result.value = 0.0;
        result.uncertainty = 0.0;
        result.valid = false;
    }
    
    return result;
}

std::vector<MeasurementResult> Metrology::measureCriticalDimensions(const cv::Mat& image, 
                                                                     const std::vector<cv::Point2f>& locations) {
    std::vector<MeasurementResult> results;
    
    for (const auto& location : locations) {
        results.push_back(measureCriticalDimension(image, location));
    }
    
    return results;
}

MeasurementResult Metrology::measureLineWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end) {
    MeasurementResult result;
    result.type = MeasurementType::LINE_WIDTH;
    result.location = (start + end) / 2.0f;
    result.unit = "nm";
    
    cv::Mat processed = preprocessForMetrology(image);
    std::vector<double> profile = extractLineProfile(processed, start, end);
    
    // Measure width using edge detection
    double threshold = cv::mean(processed).val[0];
    int edgeCount = 0;
    double totalWidth = 0.0;
    
    for (size_t i = 1; i < profile.size(); ++i) {
        if ((profile[i] > threshold && profile[i-1] <= threshold) ||
            (profile[i] < threshold && profile[i-1] >= threshold)) {
            edgeCount++;
        }
    }
    
    if (edgeCount >= 2) {
        double pixelWidth = static_cast<double>(profile.size()) / (edgeCount / 2.0);
        result.value = pixelWidth * m_calibrationFactor;
        result.uncertainty = m_measurementTolerance * result.value;
        result.valid = true;
    } else {
        result.value = 0.0;
        result.uncertainty = 0.0;
        result.valid = false;
    }
    
    return result;
}

MeasurementResult Metrology::measureSpaceWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end) {
    MeasurementResult result;
    result.type = MeasurementType::SPACE_WIDTH;
    result.location = (start + end) / 2.0f;
    result.unit = "nm";
    
    cv::Mat processed = preprocessForMetrology(image);
    std::vector<double> profile = extractLineProfile(processed, start, end);
    
    // Invert for space measurement
    cv::Mat inverted;
    cv::bitwise_not(processed, inverted);
    
    double threshold = cv::mean(inverted).val[0];
    int edgeCount = 0;
    
    for (size_t i = 1; i < profile.size(); ++i) {
        if ((profile[i] > threshold && profile[i-1] <= threshold) ||
            (profile[i] < threshold && profile[i-1] >= threshold)) {
            edgeCount++;
        }
    }
    
    if (edgeCount >= 2) {
        double pixelWidth = static_cast<double>(profile.size()) / (edgeCount / 2.0);
        result.value = pixelWidth * m_calibrationFactor;
        result.uncertainty = m_measurementTolerance * result.value;
        result.valid = true;
    } else {
        result.value = 0.0;
        result.uncertainty = 0.0;
        result.valid = false;
    }
    
    return result;
}

MeasurementResult Metrology::measureOverlayError(const cv::Mat& currentLayer, const cv::Mat& referenceLayer) {
    MeasurementResult result;
    result.type = MeasurementType::OVERLAY_ERROR;
    result.location = cv::Point2f(currentLayer.cols / 2.0f, currentLayer.rows / 2.0f);
    result.unit = "nm";
    
    // Compute phase correlation for alignment
    cv::Mat processedCurrent = preprocessForMetrology(currentLayer);
    cv::Mat processedReference = preprocessForMetrology(referenceLayer);
    
    cv::Mat correlation;
    cv::phaseCorrelate(processedCurrent, processedReference, cv::noArray(), correlation);
    
    double shiftX = correlation.at<double>(0, 0);
    double shiftY = correlation.at<double>(0, 1);
    
    result.value = std::sqrt(shiftX * shiftX + shiftY * shiftY) * m_calibrationFactor;
    result.uncertainty = m_measurementTolerance * result.value;
    result.valid = true;
    
    return result;
}

MeasurementResult Metrology::calculateFocusMetric(const cv::Mat& image) {
    MeasurementResult result;
    result.type = MeasurementType::FOCUS_METRIC;
    result.location = cv::Point2f(image.cols / 2.0f, image.rows / 2.0f);
    result.unit = "a.u.";
    
    double sharpness = calculateSharpnessMetric(image);
    double contrast = calculateContrastMetric(image);
    
    result.value = (sharpness + contrast) / 2.0;
    result.uncertainty = 0.05;
    result.valid = true;
    
    return result;
}

double Metrology::calculateSharpnessMetric(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    return stddev.val[0] * stddev.val[0];
}

double Metrology::calculateContrastMetric(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    return stddev.val[0];
}

MeasurementResult Metrology::estimateThickness(const cv::Mat& interferogram) {
    MeasurementResult result;
    result.type = MeasurementType::THICKNESS;
    result.location = cv::Point2f(interferogram.cols / 2.0f, interferogram.rows / 2.0f);
    result.unit = "nm";
    
    cv::Mat gray;
    if (interferogram.channels() == 3) {
        cv::cvtColor(interferogram, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = interferogram.clone();
    }
    
    // Simple phase analysis
    cv::Mat fft = cv::Mat::zeros(gray.size(), CV_32F);
    cv::dft(gray, fft);
    
    double maxVal = 0;
    cv::Point maxLoc;
    cv::minMaxLoc(fft, nullptr, &maxVal, nullptr, &maxLoc);
    
    // Estimate thickness from phase
    double phase = std::atan2(fft.at<float>(maxLoc.y, maxLoc.x), 0.0);
    result.value = phase * m_calibrationFactor * 1000.0; // Convert to nm
    result.uncertainty = m_measurementTolerance * result.value;
    result.valid = true;
    
    return result;
}

MeasurementResult Metrology::measureRoughness(const cv::Mat& surfaceImage) {
    MeasurementResult result;
    result.type = MeasurementType::ROUGHNESS;
    result.location = cv::Point2f(surfaceImage.cols / 2.0f, surfaceImage.rows / 2.0f);
    result.unit = "nm";
    
    cv::Mat gray;
    if (surfaceImage.channels() == 3) {
        cv::cvtColor(surfaceImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = surfaceImage.clone();
    }
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    result.value = stddev.val[0] * m_calibrationFactor;
    result.uncertainty = m_measurementTolerance * result.value;
    result.valid = true;
    
    return result;
}

std::vector<cv::Point2f> Metrology::detectEdges(const cv::Mat& image, const cv::Rect& roi) {
    cv::Mat roiImg = image(roi);
    cv::Mat edges;
    cv::Canny(roiImg, edges, 50, 150);
    
    std::vector<cv::Point> points;
    cv::findNonZero(edges, points);
    
    std::vector<cv::Point2f> edgePoints;
    for (const auto& pt : points) {
        edgePoints.push_back(cv::Point2f(pt.x + roi.x, pt.y + roi.y));
    }
    
    return edgePoints;
}

cv::Point2f Metrology::findEdgePosition(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& direction) {
    cv::Point2f current = start;
    cv::Point2f step = direction * 0.1f;
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    double threshold = cv::mean(gray).val[0];
    
    while (current.x >= 0 && current.x < image.cols && 
           current.y >= 0 && current.y < image.rows) {
        if (gray.at<uchar>(static_cast<int>(current.y), static_cast<int>(current.x)) > threshold) {
            return refineEdgeSubpixel(gray, current);
        }
        current += step;
    }
    
    return start;
}

cv::Point2f Metrology::refineEdgeSubpixel(const cv::Mat& image, const cv::Point2f& edge) {
    // Simple sub-pixel refinement using interpolation
    int x = static_cast<int>(edge.x);
    int y = static_cast<int>(edge.y);
    
    if (x > 0 && x < image.cols - 1 && y > 0 && y < image.rows - 1) {
        float dx = (image.at<uchar>(y, x+1) - image.at<uchar>(y, x-1)) / 2.0f;
        float dy = (image.at<uchar>(y+1, x) - image.at<uchar>(y-1, x)) / 2.0f;
        
        return cv::Point2f(edge.x + dx * 0.1f, edge.y + dy * 0.1f);
    }
    
    return edge;
}

void Metrology::setCalibrationFactor(double factor) {
    m_calibrationFactor = factor;
}

void Metrology::setMeasurementTolerance(double tolerance) {
    m_measurementTolerance = tolerance;
}

cv::Mat Metrology::preprocessForMetrology(const cv::Mat& image) {
    cv::Mat processed;
    cv::GaussianBlur(image, processed, cv::Size(3, 3), 0);
    return processed;
}

double Metrology::calculateLineProfile(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end) {
    std::vector<double> profile = extractLineProfile(image, start, end);
    
    if (profile.empty()) return 0.0;
    
    double sum = 0.0;
    for (double val : profile) {
        sum += val;
    }
    
    return sum / profile.size();
}

std::vector<double> Metrology::extractLineProfile(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end) {
    std::vector<double> profile;
    
    cv::LineIterator it(image, start, end);
    for (int i = 0; i < it.count; ++i, ++it) {
        profile.push_back(static_cast<double>(*it.ptr));
    }
    
    return profile;
}

double Metrology::fitGaussian(const std::vector<double>& profile) {
    // Simplified Gaussian fit
    if (profile.size() < 3) return 0.0;
    
    double maxVal = *std::max_element(profile.begin(), profile.end());
    double sum = 0.0;
    for (double val : profile) {
        sum += val;
    }
    
    return maxVal / (sum / profile.size());
}

} // namespace WaferVision

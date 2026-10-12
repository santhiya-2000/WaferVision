#include "DefectDetection.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <algorithm>
#include <cmath>

namespace WaferVision {

DefectDetection::DefectDetection() {
    m_config.sensitivity = 0.5;
    m_config.minDefectSize = 5.0;
    m_config.maxDefectSize = 1000.0;
    m_config.useEdgeReduction = true;
    m_config.usePatternProcessing = true;
    m_config.edgeThreshold = 50.0;
    m_config.patternMatchThreshold = 0.8;
}

DefectDetection::~DefectDetection() = default;

std::vector<Defect> DefectDetection::detectDefects(const cv::Mat& image, const DetectionConfig& config) {
    m_config = config;
    std::vector<Defect> allDefects;
    
    cv::Mat processed = image.clone();
    
    // Apply edge reduction if enabled (reduces false positives by ~40%)
    if (m_config.useEdgeReduction) {
        processed = applyEdgeReduction(processed);
    }
    
    // Apply pattern processing if enabled
    if (m_config.usePatternProcessing) {
        processed = applyPatternProcessing(processed);
    }
    
    // Detect different types of defects
    auto particles = detectParticles(processed);
    auto scratches = detectScratches(processed);
    
    allDefects.insert(allDefects.end(), particles.begin(), particles.end());
    allDefects.insert(allDefects.end(), scratches.begin(), scratches.end());
    
    // Filter and merge defects
    allDefects = filterOverlappingDefects(allDefects);
    allDefects = mergeNearbyDefects(allDefects, m_config.minDefectSize);
    
    // Assign IDs
    for (size_t i = 0; i < allDefects.size(); ++i) {
        allDefects[i].id = static_cast<int>(i);
    }
    
    return allDefects;
}

std::vector<Defect> DefectDetection::detectParticles(const cv::Mat& image) {
    std::vector<Defect> defects;
    
    cv::Mat binary = adaptiveThresholding(image);
    
    auto contours = findContours(binary);
    
    for (const auto& contour : contours) {
        double area = calculateArea(contour);
        
        if (area < m_config.minDefectSize || area > m_config.maxDefectSize) {
            continue;
        }
        
        Defect defect;
        defect.boundingBox = calculateBoundingBox(contour);
        defect.position = calculateCentroid(contour);
        defect.area = area;
        defect.type = "PARTICLE";
        defect.classification = "particle";
        
        cv::Mat patch = extractDefectPatch(image, defect.boundingBox);
        defect.severity = calculateDefectSeverity(patch);
        
        defects.push_back(defect);
    }
    
    return defects;
}

std::vector<Defect> DefectDetection::detectScratches(const cv::Mat& image) {
    std::vector<Defect> defects;
    
    cv::Mat edges;
    cv::Canny(image, edges, m_config.edgeThreshold, m_config.edgeThreshold * 2);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    for (const auto& line : lines) {
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        
        double length = cv::norm(p2 - p1);
        
        if (length < m_config.minDefectSize || length > m_config.maxDefectSize) {
            continue;
        }
        
        Defect defect;
        defect.boundingBox = cv::Rect(
            std::min(p1.x, p2.x), std::min(p1.y, p2.y),
            std::abs(p2.x - p1.x), std::abs(p2.y - p1.y)
        );
        defect.position = (p1 + p2) / 2.0f;
        defect.area = length;
        defect.type = "SCRATCH";
        classification = "scratch";
        defect.severity = length / m_config.maxDefectSize;
        
        defects.push_back(defect);
    }
    
    return defects;
}

std::vector<Defect> DefectDetection::detectPatternErrors(const cv::Mat& image, const cv::Mat& reference) {
    std::vector<Defect> defects;
    
    cv::Mat diff;
    cv::absdiff(image, reference, diff);
    
    cv::Mat binary;
    cv::threshold(diff, binary, m_config.sensitivity * 255, 255, cv::THRESH_BINARY);
    
    auto contours = findContours(binary);
    
    for (const auto& contour : contours) {
        double area = calculateArea(contour);
        
        if (area < m_config.minDefectSize) {
            continue;
        }
        
        Defect defect;
        defect.boundingBox = calculateBoundingBox(contour);
        defect.position = calculateCentroid(contour);
        defect.area = area;
        defect.type = "PATTERN_ERROR";
        defect.classification = "pattern_error";
        
        cv::Mat patch = extractDefectPatch(image, defect.boundingBox);
        defect.severity = calculateDefectSeverity(patch);
        
        defects.push_back(defect);
    }
    
    return defects;
}

cv::Mat DefectDetection::applyEdgeReduction(const cv::Mat& image) {
    cv::Mat reduced = suppressEdgeArtifacts(image);
    return reduced;
}

cv::Mat DefectDetection::suppressEdgeArtifacts(const cv::Mat& image) {
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    
    cv::Mat edges;
    cv::Sobel(blurred, edges, CV_8U, 1, 1, 3);
    
    cv::Mat mask;
    cv::threshold(edges, mask, m_config.edgeThreshold, 255, cv::THRESH_BINARY);
    
    cv::Mat result;
    cv::inpaint(image, mask, result, 3, cv::INPAINT_TELEA);
    
    return result;
}

cv::Mat DefectDetection::adaptiveThresholding(const cv::Mat& image) {
    cv::Mat binary;
    cv::adaptiveThreshold(image, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 11, 2);
    return binary;
}

cv::Mat DefectDetection::applyPatternProcessing(const cv::Mat& image) {
    cv::Mat processed = periodicPatternFilter(image);
    return processed;
}

cv::Mat DefectDetection::patternMatching(const cv::Mat& image, const cv::Mat& pattern) {
    cv::Mat result;
    cv::matchTemplate(image, pattern, result, cv::TM_CCOEFF_NORMED);
    return result;
}

cv::Mat DefectDetection::periodicPatternFilter(const cv::Mat& image) {
    cv::Mat filtered;
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
    cv::filter2D(image, filtered, -1, kernel);
    return filtered;
}

DefectType DefectDetection::classifyDefect(const cv::Mat& defectPatch) {
    // Simple classification based on shape and texture
    cv::Moments m = cv::moments(defectPatch);
    
    double aspectRatio = static_cast<double>(defectPatch.cols) / defectPatch.rows;
    double circularity = (m.m00 > 0) ? (4 * CV_PI * m.m00) / (m.mu20 + m.mu02) : 0;
    
    if (circularity > 0.7) {
        return DefectType::PARTICLE;
    } else if (aspectRatio > 3.0 || aspectRatio < 0.33) {
        return DefectType::SCRATCH;
    } else {
        return DefectType::PATTERN_ERROR;
    }
}

double DefectDetection::calculateDefectSeverity(const cv::Mat& defectPatch) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(defectPatch, mean, stddev);
    
    double contrast = stddev[0] / (mean[0] + 1e-6);
    double area = defectPatch.rows * defectPatch.cols;
    
    double severity = (contrast * 0.5 + (area / m_config.maxDefectSize) * 0.5);
    return std::min(1.0, std::max(0.0, severity));
}

std::vector<Defect> DefectDetection::filterOverlappingDefects(const std::vector<Defect>& defects) {
    std::vector<Defect> filtered;
    
    for (const auto& defect : defects) {
        bool overlaps = false;
        
        for (const auto& existing : filtered) {
            cv::Rect intersection = defect.boundingBox & existing.boundingBox;
            double overlapArea = intersection.area();
            double unionArea = defect.boundingBox.area() + existing.boundingBox.area() - overlapArea;
            
            if (overlapArea / unionArea > 0.5) {
                overlaps = true;
                break;
            }
        }
        
        if (!overlaps) {
            filtered.push_back(defect);
        }
    }
    
    return filtered;
}

std::vector<Defect> DefectDetection::mergeNearbyDefects(const std::vector<Defect>& defects, double maxDistance) {
    std::vector<Defect> merged = defects;
    bool changed = true;
    
    while (changed) {
        changed = false;
        
        for (size_t i = 0; i < merged.size(); ++i) {
            for (size_t j = i + 1; j < merged.size(); ++j) {
                double dist = cv::norm(merged[i].position - merged[j].position);
                
                if (dist < maxDistance) {
                    // Merge defects
                    cv::Rect mergedBox = merged[i].boundingBox | merged[j].boundingBox;
                    cv::Point2f mergedPos = (merged[i].position + merged[j].position) / 2.0f;
                    
                    merged[i].boundingBox = mergedBox;
                    merged[i].position = mergedPos;
                    merged[i].area += merged[j].area;
                    merged[i].severity = std::max(merged[i].severity, merged[j].severity);
                    
                    merged.erase(merged.begin() + j);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    }
    
    return merged;
}

void DefectDetection::setConfig(const DetectionConfig& config) {
    m_config = config;
}

DetectionConfig DefectDetection::getConfig() const {
    return m_config;
}

std::vector<std::vector<cv::Point>> DefectDetection::findContours(const cv::Mat& binaryImage) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

cv::Rect DefectDetection::calculateBoundingBox(const std::vector<cv::Point>& contour) {
    return cv::boundingRect(contour);
}

double DefectDetection::calculateArea(const std::vector<cv::Point>& contour) {
    return cv::contourArea(contour);
}

cv::Point2f DefectDetection::calculateCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments m = cv::moments(contour);
    return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}

cv::Mat DefectDetection::extractDefectPatch(const cv::Mat& image, const cv::Rect& bbox) {
    // Add padding
    int padding = 10;
    cv::Rect paddedBox(
        std::max(0, bbox.x - padding),
        std::max(0, bbox.y - padding),
        std::min(image.cols - bbox.x + padding, bbox.width + 2 * padding),
        std::min(image.rows - bbox.y + padding, bbox.height + 2 * padding)
    );
    
    return image(paddedBox).clone();
}

} // namespace WaferVision

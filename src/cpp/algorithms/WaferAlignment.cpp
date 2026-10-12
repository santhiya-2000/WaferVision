#include "WaferAlignment.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace WaferVision {

WaferAlignment::WaferAlignment() 
    : m_expectedDiameter(300.0), m_tolerance(5.0), m_method(AlignmentMethod::HYBRID) {
}

WaferAlignment::~WaferAlignment() = default;

AlignmentResult WaferAlignment::alignWafer(const cv::Mat& image, AlignmentMethod method) {
    switch (method) {
        case AlignmentMethod::CENTER_OF_MASS:
            return alignByCenterOfMass(image);
        case AlignmentMethod::HOUGH_CIRCLE:
            return alignByHoughCircle(image);
        case AlignmentMethod::TEMPLATE_MATCHING:
            return alignByTemplateMatching(image, createWaferTemplate(static_cast<int>(m_expectedDiameter)));
        case AlignmentMethod::EDGE_BASED:
            return alignByEdgeDetection(image);
        case AlignmentMethod::HYBRID:
            return alignHybrid(image);
        default:
            return alignHybrid(image);
    }
}

AlignmentResult WaferAlignment::alignByCenterOfMass(const cv::Mat& image) {
    AlignmentResult result;
    
    cv::Mat processed = preprocessForAlignment(image);
    cv::Point2f center = findCenterOfMass(processed);
    
    double rotation = calculateFlatAngle(image, center);
    
    result.center = center;
    result.rotation = rotation;
    result.confidence = 0.7;
    result.success = true;
    
    cv::Mat translation = (cv::Mat_<double>(3, 3) << 
        1, 0, image.cols/2.0 - center.x,
        0, 1, image.rows/2.0 - center.y,
        0, 0, 1);
    
    cv::Mat rotationMat = cv::getRotationMatrix2D(
        cv::Point2f(image.cols/2.0, image.rows/2.0), 
        -rotation, 1.0);
    
    cv::Mat rotExtended = (cv::Mat_<double>(3, 3) << 
        rotationMat.at<double>(0, 0), rotationMat.at<double>(0, 1), rotationMat.at<double>(0, 2),
        rotationMat.at<double>(1, 0), rotationMat.at<double>(1, 1), rotationMat.at<double>(1, 2),
        0, 0, 1);
    
    result.transformationMatrix = rotExtended * translation;
    
    return result;
}

AlignmentResult WaferAlignment::alignByHoughCircle(const cv::Mat& image) {
    AlignmentResult result;
    
    cv::Mat processed = preprocessForAlignment(image);
    std::vector<cv::Vec3f> circles = detectCircles(processed);
    
    if (circles.empty()) {
        result.success = false;
        return result;
    }
    
    cv::Vec3f bestCircle = circles[0];
    double minDiff = std::abs(bestCircle[2] - m_expectedDiameter / 2.0);
    
    for (const auto& circle : circles) {
        double diff = std::abs(circle[2] - m_expectedDiameter / 2.0);
        if (diff < minDiff) {
            minDiff = diff;
            bestCircle = circle;
        }
    }
    
    cv::Point2f center(bestCircle[0], bestCircle[1]);
    double rotation = calculateFlatAngle(image, center);
    
    result.center = center;
    result.rotation = rotation;
    result.confidence = 0.9;
    result.success = true;
    
    return result;
}

AlignmentResult WaferAlignment::alignByTemplateMatching(const cv::Mat& image, const cv::Mat& templateImage) {
    AlignmentResult result;
    
    cv::Mat processed = preprocessForAlignment(image);
    
    double bestScore = 0.0;
    cv::Point bestLocation;
    double bestScale = 1.0;
    
    for (double scale = 0.9; scale <= 1.1; scale += 0.05) {
        cv::Mat scaledTemplate;
        cv::resize(templateImage, scaledTemplate, cv::Size(), scale, scale);
        
        if (scaledTemplate.rows > processed.rows || scaledTemplate.cols > processed.cols) {
            continue;
        }
        
        cv::Mat resultMatrix;
        cv::matchTemplate(processed, scaledTemplate, resultMatrix, cv::TM_CCOEFF_NORMED);
        
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(resultMatrix, &minVal, &maxVal, &minLoc, &maxLoc);
        
        if (maxVal > bestScore) {
            bestScore = maxVal;
            bestLocation = maxLoc;
            bestScale = scale;
        }
    }
    
    if (bestScore < 0.5) {
        result.success = false;
        return result;
    }
    
    cv::Point2f center(bestLocation.x + templateImage.cols * bestScale / 2.0,
                       bestLocation.y + templateImage.rows * bestScale / 2.0);
    
    double rotation = calculateFlatAngle(image, center);
    
    result.center = center;
    result.rotation = rotation;
    result.confidence = bestScore;
    result.success = true;
    
    return result;
}

AlignmentResult WaferAlignment::alignByEdgeDetection(const cv::Mat& image) {
    AlignmentResult result;
    
    cv::Mat processed = preprocessForAlignment(image);
    cv::Mat edges;
    cv::Canny(processed, edges, 50, 150);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 
                    processed.rows/8, 100, 30, 
                    m_expectedDiameter/2 - m_tolerance, 
                    m_expectedDiameter/2 + m_tolerance);
    
    if (circles.empty()) {
        result.success = false;
        return result;
    }
    
    cv::Point2f center(circles[0][0], circles[0][1]);
    double rotation = calculateFlatAngle(image, center);
    
    result.center = center;
    result.rotation = rotation;
    result.confidence = 0.85;
    result.success = true;
    
    return result;
}

AlignmentResult WaferAlignment::alignHybrid(const cv::Mat& image) {
    // Try multiple methods and combine results
    auto houghResult = alignByHoughCircle(image);
    auto centerResult = alignByCenterOfMass(image);
    
    if (houghResult.success && houghResult.confidence > 0.8) {
        return houghResult;
    }
    
    if (centerResult.success) {
        return centerResult;
    }
    
    return alignByEdgeDetection(image);
}

cv::Mat WaferAlignment::applyAlignment(const cv::Mat& image, const AlignmentResult& result) {
    if (!result.success) {
        return image.clone();
    }
    
    cv::Mat aligned;
    cv::warpPerspective(image, aligned, result.transformationMatrix, image.size());
    return aligned;
}

cv::Point2f WaferAlignment::detectFlat(const cv::Mat& image) {
    cv::Mat processed = preprocessForAlignment(image);
    cv::Point2f center = findCenterOfMass(processed);
    
    // Search for flat edge
    cv::Mat edges;
    cv::Canny(processed, edges, 50, 150);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    
    cv::Point2f flatMidpoint;
    double maxDist = 0.0;
    
    for (const auto& line : lines) {
        cv::Point2f p1(line[0], line[1]);
        cv::Point2f p2(line[2], line[3]);
        cv::Point2f midpoint = (p1 + p2) / 2.0f;
        
        double dist = cv::norm(midpoint - center);
        if (dist > maxDist) {
            maxDist = dist;
            flatMidpoint = midpoint;
        }
    }
    
    return flatMidpoint;
}

double WaferAlignment::calculateFlatAngle(const cv::Mat& image, const cv::Point2f& center) {
    cv::Point2f flat = detectFlat(image);
    cv::Point2f direction = flat - center;
    return std::atan2(direction.y, direction.x) * 180.0 / CV_PI;
}

cv::Point2f WaferAlignment::detectNotch(const cv::Mat& image) {
    cv::Mat processed = preprocessForAlignment(image);
    cv::Point2f center = findCenterOfMass(processed);
    
    // Find notch using contour analysis
    cv::Mat edges;
    cv::Canny(processed, edges, 50, 150);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Point2f notchPosition;
    double maxDist = 0.0;
    
    for (const auto& contour : contours) {
        cv::Moments m = cv::moments(contour);
        if (m.m00 > 0) {
            cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
            double dist = cv::norm(centroid - center);
            if (dist > maxDist && dist < m_expectedDiameter / 2.0) {
                maxDist = dist;
                notchPosition = centroid;
            }
        }
    }
    
    return notchPosition;
}

double WaferVision::WaferAlignment::calculateNotchAngle(const cv::Mat& image, const cv::Point2f& center) {
    cv::Point2f notch = detectNotch(image);
    cv::Point2f direction = notch - center;
    return std::atan2(direction.y, direction.x) * 180.0 / CV_PI;
}

void WaferAlignment::setExpectedDiameter(double diameter) {
    m_expectedDiameter = diameter;
}

void WaferAlignment::setTolerance(double tolerance) {
    m_tolerance = tolerance;
}

void WaferAlignment::setAlignmentMethod(AlignmentMethod method) {
    m_method = method;
}

cv::Mat WaferAlignment::preprocessForAlignment(const cv::Mat& image) {
    cv::Mat processed;
    cv::GaussianBlur(image, processed, cv::Size(5, 5), 0);
    cv::threshold(processed, processed, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    return processed;
}

cv::Point2f WaferAlignment::findCenterOfMass(const cv::Mat& binaryImage) {
    cv::Moments m = cv::moments(binaryImage);
    return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}

std::vector<cv::Vec3f> WaferAlignment::detectCircles(const cv::Mat& image) {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT, 1, 
                    image.rows/8, 100, 30, 
                    m_expectedDiameter/2 - m_tolerance, 
                    m_expectedDiameter/2 + m_tolerance);
    return circles;
}

cv::Mat WaferAlignment::createWaferTemplate(int diameter) {
    cv::Mat templateImg(diameter + 20, diameter + 20, CV_8U, cv::Scalar(0));
    cv::Point center(diameter/2 + 10, diameter/2 + 10);
    cv::circle(templateImg, center, diameter/2, cv::Scalar(255), -1);
    
    // Add flat
    cv::Point flatStart(center.x - diameter/2, center.y);
    cv::Point flatEnd(center.x - diameter/2 + diameter/10, center.y);
    cv::line(templateImg, flatStart, flatEnd, cv::Scalar(0), 3);
    
    return templateImg;
}

} // namespace WaferVision

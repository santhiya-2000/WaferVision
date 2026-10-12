#include "WaferData.hpp"
#include <fstream>
#include <sstream>

namespace WaferVision {

WaferData::WaferData() 
    : m_waferDiameter(300.0) { // Default 300mm wafer
    m_waferMap.center = cv::Point2f(0, 0);
    m_waferMap.radius = m_waferDiameter / 2.0;
    m_waferMap.rotation = 0.0;
}

WaferData::WaferData(const std::string& imagePath) 
    : WaferData() {
    loadImage(imagePath);
}

WaferData::~WaferData() = default;

void WaferData::loadImage(const std::string& path) {
    m_waferMap.image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (m_waferMap.image.empty()) {
        throw std::runtime_error("Failed to load wafer image: " + path);
    }
    
    // Initialize center to image center
    m_waferMap.center = cv::Point2f(
        m_waferMap.image.cols / 2.0f,
        m_waferMap.image.rows / 2.0f
    );
    
    // Initialize alignment matrix to identity
    m_waferMap.alignmentMatrix = cv::Mat::eye(3, 3, CV_64F);
}

cv::Mat WaferData::getImage() const {
    return m_waferMap.image;
}

WaferMap& WaferData::getWaferMap() {
    return m_waferMap;
}

const WaferMap& WaferData::getWaferMap() const {
    return m_waferMap;
}

MetrologyData& WaferData::getMetrologyData() {
    return m_metrologyData;
}

const MetrologyData& WaferData::getMetrologyData() const {
    return m_metrologyData;
}

void WaferData::setWaferDiameter(double diameter) {
    m_waferDiameter = diameter;
    m_waferMap.radius = diameter / 2.0;
}

double WaferData::getWaferDiameter() const {
    return m_waferDiameter;
}

void WaferData::setWaferCenter(const cv::Point2f& center) {
    m_waferMap.center = center;
}

cv::Point2f WaferData::getWaferCenter() const {
    return m_waferMap.center;
}

void WaferData::addDefect(const Defect& defect) {
    m_waferMap.defects.push_back(defect);
}

void WaferData::clearDefects() {
    m_waferMap.defects.clear();
}

const std::vector<Defect>& WaferData::getDefects() const {
    return m_waferMap.defects;
}

int WaferData::getDefectCount() const {
    return static_cast<int>(m_waferMap.defects.size());
}

void WaferData::setMetadata(const std::string& key, const std::string& value) {
    m_metadata[key] = value;
}

std::string WaferData::getMetadata(const std::string& key) const {
    auto it = m_metadata.find(key);
    if (it != m_metadata.end()) {
        return it->second;
    }
    return "";
}

} // namespace WaferVision

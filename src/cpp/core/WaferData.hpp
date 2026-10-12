#ifndef WAFER_DATA_HPP
#define WAFER_DATA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace WaferVision {

struct Defect {
    int id;
    cv::Point2f position;
    cv::Rect boundingBox;
    double area;
    double severity;
    std::string type;
    std::string classification;
};

struct WaferMap {
    cv::Mat image;
    std::vector<Defect> defects;
    cv::Point2f center;
    double radius;
    double rotation;
    cv::Mat alignmentMatrix;
};

struct MetrologyData {
    double criticalDimension;
    double lineWidth;
    double spaceWidth;
    double overlayError;
    double focusMetric;
    std::vector<cv::Point2f> measurementPoints;
};

class WaferData {
public:
    WaferData();
    explicit WaferData(const std::string& imagePath);
    ~WaferData();

    // Data access
    void loadImage(const std::string& path);
    cv::Mat getImage() const;
    WaferMap& getWaferMap();
    const WaferMap& getWaferMap() const;
    MetrologyData& getMetrologyData();
    const MetrologyData& getMetrologyData() const;

    // Wafer properties
    void setWaferDiameter(double diameter);
    double getWaferDiameter() const;
    void setWaferCenter(const cv::Point2f& center);
    cv::Point2f getWaferCenter() const;

    // Defect management
    void addDefect(const Defect& defect);
    void clearDefects();
    const std::vector<Defect>& getDefects() const;
    int getDefectCount() const;

    // Metadata
    void setMetadata(const std::string& key, const std::string& value);
    std::string getMetadata(const std::string& key) const;

private:
    WaferMap m_waferMap;
    MetrologyData m_metrologyData;
    double m_waferDiameter;
    std::map<std::string, std::string> m_metadata;
};

} // namespace WaferVision

#endif // WAFER_DATA_HPP

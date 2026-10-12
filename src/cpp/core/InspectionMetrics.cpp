#include "InspectionMetrics.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace WaferVision {

InspectionMetrics::InspectionMetrics()
    : m_totalProcessingTime(0.0), m_wafersProcessed(0), 
      m_totalDefectsDetected(0), m_falsePositives(0) {
}

InspectionMetrics::~InspectionMetrics() = default;

void InspectionMetrics::recordMetric(const std::string& name, double value, const std::string& unit) {
    MetricValue mv;
    mv.value = value;
    mv.unit = unit;
    mv.timestamp = std::chrono::system_clock::now();
    
    m_metrics[name].push_back(mv);
}

void InspectionMetrics::recordTiming(const std::string& operation, double milliseconds) {
    m_timings[operation] = milliseconds;
    m_totalProcessingTime += milliseconds;
}

double InspectionMetrics::getMetric(const std::string& name) const {
    auto it = m_metrics.find(name);
    if (it != m_metrics.end() && !it->second.empty()) {
        return it->second.back().value;
    }
    return 0.0;
}

std::string InspectionMetrics::getMetricUnit(const std::string& name) const {
    auto it = m_metrics.find(name);
    if (it != m_metrics.end() && !it->second.empty()) {
        return it->second.back().unit;
    }
    return "";
}

std::vector<std::string> InspectionMetrics::getMetricNames() const {
    std::vector<std::string> names;
    for (const auto& pair : m_metrics) {
        names.push_back(pair.first);
    }
    return names;
}

double InspectionMetrics::getProcessingTime() const {
    return m_totalProcessingTime;
}

double InspectionMetrics::getThroughput() const {
    if (m_totalProcessingTime == 0.0) return 0.0;
    return (m_wafersProcessed / m_totalProcessingTime) * 3600000.0; // wafers per hour
}

double InspectionMetrics::getDefectDetectionRate() const {
    if (m_wafersProcessed == 0) return 0.0;
    return static_cast<double>(m_totalDefectsDetected) / m_wafersProcessed;
}

double InspectionMetrics::getFalsePositiveRate() const {
    if (m_totalDefectsDetected == 0) return 0.0;
    return static_cast<double>(m_falsePositives) / m_totalDefectsDetected;
}

void InspectionMetrics::calculateStatistics() {
    // Statistics calculated on-demand in getMean, getStdDev, etc.
}

double InspectionMetrics::getMean(const std::string& metricName) const {
    auto it = m_metrics.find(metricName);
    if (it == m_metrics.end() || it->second.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& mv : it->second) {
        sum += mv.value;
    }
    return sum / it->second.size();
}

double InspectionMetrics::getStdDev(const std::string& metricName) const {
    auto it = m_metrics.find(metricName);
    if (it == m_metrics.end() || it->second.size() < 2) return 0.0;
    
    double mean = getMean(metricName);
    double sumSqDiff = 0.0;
    
    for (const auto& mv : it->second) {
        double diff = mv.value - mean;
        sumSqDiff += diff * diff;
    }
    
    return std::sqrt(sumSqDiff / it->second.size());
}

double InspectionMetrics::getMin(const std::string& metricName) const {
    auto it = m_metrics.find(metricName);
    if (it == m_metrics.end() || it->second.empty()) return 0.0;
    
    double minVal = it->second[0].value;
    for (const auto& mv : it->second) {
        if (mv.value < minVal) minVal = mv.value;
    }
    return minVal;
}

double InspectionMetrics::getMax(const std::string& metricName) const {
    auto it = m_metrics.find(metricName);
    if (it == m_metrics.end() || it->second.empty()) return 0.0;
    
    double maxVal = it->second[0].value;
    for (const auto& mv : it->second) {
        if (mv.value > maxVal) maxVal = mv.value;
    }
    return maxVal;
}

std::string InspectionMetrics::generateReport() const {
    std::stringstream ss;
    
    ss << "=== WaferVision Inspection Report ===\n\n";
    ss << "Performance Metrics:\n";
    ss << "  Total Processing Time: " << std::fixed << std::setprecision(2) 
       << m_totalProcessingTime << " ms\n";
    ss << "  Wafers Processed: " << m_wafersProcessed << "\n";
    ss << "  Throughput: " << std::fixed << std::setprecision(2) 
       << getThroughput() << " wafers/hour\n";
    ss << "  Defect Detection Rate: " << std::fixed << std::setprecision(2) 
       << getDefectDetectionRate() << " defects/wafer\n";
    ss << "  False Positive Rate: " << std::fixed << std::setprecision(2) 
       << (getFalsePositiveRate() * 100.0) << "%\n\n";
    
    ss << "Timing Breakdown:\n";
    for (const auto& pair : m_timings) {
        ss << "  " << pair.first << ": " << std::fixed << std::setprecision(2) 
           << pair.second << " ms\n";
    }
    
    ss << "\nDetailed Metrics:\n";
    for (const auto& pair : m_metrics) {
        ss << "  " << pair.first << ": " << std::fixed << std::setprecision(4) 
           << getMean(pair.first) << " " << getMetricUnit(pair.first) 
           << " (±" << getStdDev(pair.first) << ")\n";
    }
    
    return ss.str();
}

void InspectionMetrics::exportToCsv(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }
    
    file << "Metric,Value,Unit,Timestamp\n";
    for (const auto& pair : m_metrics) {
        for (const auto& mv : pair.second) {
            auto time_t = std::chrono::system_clock::to_time_t(mv.timestamp);
            file << pair.first << "," << mv.value << "," << mv.unit << "," 
                 << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
        }
    }
    
    file.close();
}

void InspectionMetrics::exportToJson(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }
    
    file << "{\n";
    file << "  \"performance\": {\n";
    file << "    \"totalProcessingTime\": " << m_totalProcessingTime << ",\n";
    file << "    \"wafersProcessed\": " << m_wafersProcessed << ",\n";
    file << "    \"throughput\": " << getThroughput() << ",\n";
    file << "    \"defectDetectionRate\": " << getDefectDetectionRate() << ",\n";
    file << "    \"falsePositiveRate\": " << getFalsePositiveRate() << "\n";
    file << "  },\n";
    file << "  \"timings\": {\n";
    
    bool first = true;
    for (const auto& pair : m_timings) {
        if (!first) file << ",\n";
        file << "    \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    
    file << "\n  },\n";
    file << "  \"metrics\": {\n";
    
    first = true;
    for (const auto& pair : m_metrics) {
        if (!first) file << ",\n";
        file << "    \"" << pair.first << "\": {\n";
        file << "      \"mean\": " << getMean(pair.first) << ",\n";
        file << "      \"stdDev\": " << getStdDev(pair.first) << ",\n";
        file << "      \"min\": " << getMin(pair.first) << ",\n";
        file << "      \"max\": " << getMax(pair.first) << ",\n";
        file << "      \"unit\": \"" << getMetricUnit(pair.first) << "\"\n";
        file << "    }";
        first = false;
    }
    
    file << "\n  }\n";
    file << "}\n";
    file.close();
}

void InspectionMetrics::reset() {
    m_metrics.clear();
    m_timings.clear();
    m_totalProcessingTime = 0.0;
    m_wafersProcessed = 0;
    m_totalDefectsDetected = 0;
    m_falsePositives = 0;
}

} // namespace WaferVision

#ifndef INSPECTION_METRICS_HPP
#define INSPECTION_METRICS_HPP

#include <string>
#include <vector>
#include <map>
#include <chrono>

namespace WaferVision {

struct MetricValue {
    double value;
    std::string unit;
    std::chrono::system_clock::time_point timestamp;
};

class InspectionMetrics {
public:
    InspectionMetrics();
    ~InspectionMetrics();

    // Metric recording
    void recordMetric(const std::string& name, double value, const std::string& unit = "");
    void recordTiming(const std::string& operation, double milliseconds);
    
    // Metric retrieval
    double getMetric(const std::string& name) const;
    std::string getMetricUnit(const std::string& name) const;
    std::vector<std::string> getMetricNames() const;
    
    // Performance metrics
    double getProcessingTime() const;
    double getThroughput() const; // wafers per hour
    double getDefectDetectionRate() const;
    double getFalsePositiveRate() const;
    
    // Statistics
    void calculateStatistics();
    double getMean(const std::string& metricName) const;
    double getStdDev(const std::string& metricName) const;
    double getMin(const std::string& metricName) const;
    double getMax(const std::string& metricName) const;
    
    // Reporting
    std::string generateReport() const;
    void exportToCsv(const std::string& path) const;
    void exportToJson(const std::string& path) const;
    
    // Reset
    void reset();

private:
    std::map<std::string, std::vector<MetricValue>> m_metrics;
    std::map<std::string, double> m_timings;
    double m_totalProcessingTime;
    int m_wafersProcessed;
    int m_totalDefectsDetected;
    int m_falsePositives;
};

} // namespace WaferVision

#endif // INSPECTION_METRICS_HPP

#include "core/ImageProcessor.hpp"
#include "core/WaferData.hpp"
#include "core/InspectionMetrics.hpp"
#include "algorithms/WaferAlignment.hpp"
#include "algorithms/DefectDetection.hpp"
#include "algorithms/Metrology.hpp"
#include "utils/Logger.hpp"
#include "utils/ConfigManager.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <memory>

using namespace WaferVision;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --input <path>        Input wafer image path\n"
              << "  --output <path>       Output directory for results\n"
              << "  --align               Perform wafer alignment\n"
              << "  --detect              Perform defect detection\n"
              << "  --metrology           Perform metrology measurements\n"
              << "  --gpu                 Enable GPU acceleration\n"
              << "  --config <path>       Configuration file path\n"
              << "  --help                Show this help message\n";
}

int main(int argc, char** argv) {
    Logger::getInstance().setConsoleOutput(true);
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    
    LOG_INFO("WaferVision Semiconductor Inspection System");
    LOG_INFO("===========================================");
    
    std::string inputPath;
    std::string outputPath = ".";
    std::string configPath;
    bool doAlign = false;
    bool doDetect = false;
    bool doMetrology = false;
    bool useGpu = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--align") {
            doAlign = true;
        } else if (arg == "--detect") {
            doDetect = true;
        } else if (arg == "--metrology") {
            doMetrology = true;
        } else if (arg == "--gpu") {
            useGpu = true;
        } else if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Load configuration if provided
    if (!configPath.empty()) {
        ConfigManager::getInstance().loadFromFile(configPath);
        LOG_INFO("Loaded configuration from: " + configPath);
    }
    
    // Check if input is provided
    if (inputPath.empty()) {
        LOG_ERROR("No input image specified. Use --input <path>");
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Load wafer data
        LOG_INFO("Loading wafer image: " + inputPath);
        WaferData wafer(inputPath);
        
        // Initialize image processor
        ImageProcessor processor;
        processor.setUseGpu(useGpu);
        
        if (processor.isGpuAvailable() && useGpu) {
            LOG_INFO("GPU acceleration enabled");
        } else {
            LOG_INFO("Using CPU for processing");
        }
        
        // Preprocess image
        LOG_INFO("Preprocessing image...");
        cv::Mat image = wafer.getImage();
        cv::Mat processed = processor.preprocessImage(image);
        
        // Initialize metrics
        InspectionMetrics metrics;
        
        // Perform alignment if requested
        if (doAlign) {
            LOG_INFO("Performing wafer alignment...");
            auto alignStart = std::chrono::high_resolution_clock::now();
            
            WaferAlignment alignment;
            AlignmentResult result = alignment.alignWafer(processed);
            
            if (result.success) {
                processed = alignment.applyAlignment(processed, result);
                LOG_INFO("Alignment complete. Center: (" + 
                        std::to_string(result.center.x) + ", " + 
                        std::to_string(result.center.y) + ")");
                
                auto alignEnd = std::chrono::high_resolution_clock::now();
                double alignTime = std::chrono::duration<double, std::milli>(alignEnd - alignStart).count();
                metrics.recordTiming("alignment", alignTime);
            } else {
                LOG_WARNING("Alignment failed");
            }
        }
        
        // Perform defect detection if requested
        if (doDetect) {
            LOG_INFO("Performing defect detection...");
            auto detectStart = std::chrono::high_resolution_clock::now();
            
            DefectDetection detector;
            DetectionConfig config;
            config.useEdgeReduction = true;
            config.usePatternProcessing = true;
            config.sensitivity = 0.5;
            
            std::vector<Defect> defects = detector.detectDefects(processed, config);
            
            for (const auto& defect : defects) {
                wafer.addDefect(defect);
            }
            
            LOG_INFO("Defect detection complete. Found " + 
                    std::to_string(defects.size()) + " defects");
            
            auto detectEnd = std::chrono::high_resolution_clock::now();
            double detectTime = std::chrono::duration<double, std::milli>(detectEnd - detectStart).count();
            metrics.recordTiming("defect_detection", detectTime);
            metrics.recordMetric("defect_count", defects.size(), "count");
        }
        
        // Perform metrology if requested
        if (doMetrology) {
            LOG_INFO("Performing metrology measurements...");
            auto metrologyStart = std::chrono::high_resolution_clock::now();
            
            Metrology metrology;
            metrology.setCalibrationFactor(1.0);
            
            MetrologyData metroData = metrology.runMetrology(processed, wafer.getWaferMap());
            
            LOG_INFO("Metrology complete:");
            LOG_INFO("  Critical Dimension: " + std::to_string(metroData.criticalDimension));
            LOG_INFO("  Line Width: " + std::to_string(metroData.lineWidth));
            LOG_INFO("  Focus Metric: " + std::to_string(metroData.focusMetric));
            
            auto metrologyEnd = std::chrono::high_resolution_clock::now();
            double metrologyTime = std::chrono::duration<double, std::milli>(metrologyEnd - metrologyStart).count();
            metrics.recordTiming("metrology", metrologyTime);
            metrics.recordMetric("critical_dimension", metroData.criticalDimension, "nm");
            metrics.recordMetric("focus_metric", metroData.focusMetric, "a.u.");
        }
        
        // Save results
        LOG_INFO("Saving results to: " + outputPath);
        cv::imwrite(outputPath + "/processed.png", processed);
        
        // Save defect map if defects were detected
        if (wafer.getDefectCount() > 0) {
            cv::Mat defectMap = cv::Mat::zeros(processed.size(), CV_8UC3);
            cv::cvtColor(processed, defectMap, cv::COLOR_GRAY2BGR);
            
            for (const auto& defect : wafer.getDefects()) {
                cv::rectangle(defectMap, defect.boundingBox, cv::Scalar(0, 0, 255), 2);
            }
            
            cv::imwrite(outputPath + "/defect_map.png", defectMap);
        }
        
        // Generate report
        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        metrics.recordTiming("total", totalTime);
        
        std::string report = metrics.generateReport();
        LOG_INFO(report);
        
        // Save report to file
        std::ofstream reportFile(outputPath + "/inspection_report.txt");
        if (reportFile.is_open()) {
            reportFile << report;
            reportFile.close();
        }
        
        LOG_INFO("Processing complete. Total time: " + std::to_string(totalTime) + " ms");
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error: " + std::string(e.what()));
        return 1;
    }
}

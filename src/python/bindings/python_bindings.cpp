#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "core/WaferData.hpp"
#include "core/ImageProcessor.hpp"
#include "algorithms/WaferAlignment.hpp"
#include "algorithms/DefectDetection.hpp"
#include "algorithms/Metrology.hpp"

namespace py = pybind11;

PYBIND11_MODULE(wafervision_python, m) {
    m.doc() = "WaferVision Python bindings for semiconductor wafer inspection";
    
    // WaferData bindings
    py::class_<WaferVision::Defect>(m, "Defect")
        .def(py::init<>())
        .def_readwrite("id", &WaferVision::Defect::id)
        .def_readwrite("position", &WaferVision::Defect::position)
        .def_readwrite("bounding_box", &WaferVision::Defect::boundingBox)
        .def_readwrite("area", &WaferVision::Defect::area)
        .def_readwrite("severity", &WaferVision::Defect::severity)
        .def_readwrite("type", &WaferVision::Defect::type)
        .def_readwrite("classification", &WaferVision::Defect::classification);
    
    py::class_<WaferVision::WaferMap>(m, "WaferMap")
        .def(py::init<>())
        .def_readwrite("image", &WaferVision::WaferMap::image)
        .def_readwrite("defects", &WaferVision::WaferMap::defects)
        .def_readwrite("center", &WaferVision::WaferMap::center)
        .def_readwrite("radius", &WaferVision::WaferMap::radius)
        .def_readwrite("rotation", &WaferVision::WaferMap::rotation);
    
    py::class_<WaferVision::MetrologyData>(m, "MetrologyData")
        .def(py::init<>())
        .def_readwrite("critical_dimension", &WaferVision::MetrologyData::criticalDimension)
        .def_readwrite("line_width", &WaferVision::MetrologyData::lineWidth)
        .def_readwrite("space_width", &WaferVision::MetrologyData::spaceWidth)
        .def_readwrite("overlay_error", &WaferVision::MetrologyData::overlayError)
        .def_readwrite("focus_metric", &WaferVision::MetrologyData::focusMetric)
        .def_readwrite("measurement_points", &WaferVision::MetrologyData::measurementPoints);
    
    py::class_<WaferVision::WaferData>(m, "WaferData")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("load_image", &WaferVision::WaferData::loadImage)
        .def("get_image", &WaferVision::WaferData::getImage)
        .def("get_wafer_map", py::overload_cast<>(&WaferVision::WaferData::getWaferMap))
        .def("get_metrology_data", py::overload_cast<>(&WaferVision::WaferData::getMetrologyData))
        .def("set_wafer_diameter", &WaferVision::WaferData::setWaferDiameter)
        .def("get_wafer_diameter", &WaferVision::WaferData::getWaferDiameter)
        .def("set_wafer_center", &WaferVision::WaferData::setWaferCenter)
        .def("get_wafer_center", &WaferVision::WaferData::getWaferCenter)
        .def("add_defect", &WaferVision::WaferData::addDefect)
        .def("clear_defects", &WaferVision::WaferData::clearDefects)
        .def("get_defects", &WaferVision::WaferData::getDefects)
        .def("get_defect_count", &WaferVision::WaferData::getDefectCount)
        .def("set_metadata", &WaferVision::WaferData::setMetadata)
        .def("get_metadata", &WaferVision::WaferData::getMetadata);
    
    // ImageProcessor bindings
    py::class_<WaferVision::ImageProcessor>(m, "ImageProcessor")
        .def(py::init<>())
        .def("load_image", &WaferVision::ImageProcessor::loadImage)
        .def("preprocess_image", &WaferVision::ImageProcessor::preprocessImage)
        .def("normalize_image", &WaferVision::ImageProcessor::normalizeImage)
        .def("denoise_image", &WaferVision::ImageProcessor::denoiseImage)
        .def("enhance_contrast", &WaferVision::ImageProcessor::enhanceContrast)
        .def("sharpen_image", &WaferVision::ImageProcessor::sharpenImage)
        .def("adjust_brightness", &WaferVision::ImageProcessor::adjustBrightness)
        .def("rotate_image", &WaferVision::ImageProcessor::rotateImage)
        .def("scale_image", &WaferVision::ImageProcessor::scaleImage)
        .def("translate_image", &WaferVision::ImageProcessor::translateImage)
        .def("is_gpu_available", &WaferVision::ImageProcessor::isGpuAvailable)
        .def("set_use_gpu", &WaferVision::ImageProcessor::setUseGpu);
    
    // WaferAlignment bindings
    py::enum_<WaferVision::AlignmentMethod>(m, "AlignmentMethod")
        .value("CENTER_OF_MASS", WaferVision::AlignmentMethod::CENTER_OF_MASS)
        .value("HOUGH_CIRCLE", WaferVision::AlignmentMethod::HOUGH_CIRCLE)
        .value("TEMPLATE_MATCHING", WaferVision::AlignmentMethod::TEMPLATE_MATCHING)
        .value("EDGE_BASED", WaferVision::AlignmentMethod::EDGE_BASED)
        .value("HYBRID", WaferVision::AlignmentMethod::HYBRID);
    
    py::class_<WaferVision::AlignmentResult>(m, "AlignmentResult")
        .def(py::init<>())
        .def_readwrite("center", &WaferVision::AlignmentResult::center)
        .def_readwrite("rotation", &WaferVision::AlignmentResult::rotation)
        .def_readwrite("confidence", &WaferVision::AlignmentResult::confidence)
        .def_readwrite("success", &WaferVision::AlignmentResult::success);
    
    py::class_<WaferVision::WaferAlignment>(m, "WaferAlignment")
        .def(py::init<>())
        .def("align_wafer", &WaferVision::WaferAlignment::alignWafer)
        .def("align_by_center_of_mass", &WaferVision::WaferAlignment::alignByCenterOfMass)
        .def("align_by_hough_circle", &WaferVision::WaferAlignment::alignByHoughCircle)
        .def("apply_alignment", &WaferVision::WaferAlignment::applyAlignment)
        .def("set_expected_diameter", &WaferVision::WaferAlignment::setExpectedDiameter)
        .def("set_tolerance", &WaferVision::WaferAlignment::setTolerance);
    
    // DefectDetection bindings
    py::enum_<WaferVision::DefectType>(m, "DefectType")
        .value("PARTICLE", WaferVision::DefectType::PARTICLE)
        .value("SCRATCH", WaferVision::DefectType::SCRATCH)
        .value("PATTERN_ERROR", WaferVision::DefectType::PATTERN_ERROR)
        .value("MISSING_FEATURE", WaferVision::DefectType::MISSING_FEATURE)
        .value("EXCESS_MATERIAL", WaferVision::DefectType::EXCESS_MATERIAL)
        .value("UNKNOWN", WaferVision::DefectType::UNKNOWN);
    
    py::class_<WaferVision::DetectionConfig>(m, "DetectionConfig")
        .def(py::init<>())
        .def_readwrite("sensitivity", &WaferVision::DetectionConfig::sensitivity)
        .def_readwrite("min_defect_size", &WaferVision::DetectionConfig::minDefectSize)
        .def_readwrite("max_defect_size", &WaferVision::DetectionConfig::maxDefectSize)
        .def_readwrite("use_edge_reduction", &WaferVision::DetectionConfig::useEdgeReduction)
        .def_readwrite("use_pattern_processing", &WaferVision::DetectionConfig::usePatternProcessing)
        .def_readwrite("edge_threshold", &WaferVision::DetectionConfig::edgeThreshold)
        .def_readwrite("pattern_match_threshold", &WaferVision::DetectionConfig::patternMatchThreshold);
    
    py::class_<WaferVision::DefectDetection>(m, "DefectDetection")
        .def(py::init<>())
        .def("detect_defects", &WaferVision::DefectDetection::detectDefects)
        .def("detect_particles", &WaferVision::DefectDetection::detectParticles)
        .def("detect_scratches", &WaferVision::DefectDetection::detectScratches)
        .def("detect_pattern_errors", &WaferVision::DefectDetection::detectPatternErrors)
        .def("apply_edge_reduction", &WaferVision::DefectDetection::applyEdgeReduction)
        .def("set_config", &WaferVision::DefectDetection::setConfig)
        .def("get_config", &WaferVision::DefectDetection::getConfig);
    
    // Metrology bindings
    py::enum_<WaferVision::MeasurementType>(m, "MeasurementType")
        .value("CRITICAL_DIMENSION", WaferVision::MeasurementType::CRITICAL_DIMENSION)
        .value("LINE_WIDTH", WaferVision::MeasurementType::LINE_WIDTH)
        .value("SPACE_WIDTH", WaferVision::MeasurementType::SPACE_WIDTH)
        .value("OVERLAY_ERROR", WaferVision::MeasurementType::OVERLAY_ERROR)
        .value("FOCUS_METRIC", WaferVision::MeasurementType::FOCUS_METRIC)
        .value("THICKNESS", WaferVision::MeasurementType::THICKNESS)
        .value("ROUGHNESS", WaferVision::MeasurementType::ROUGHNESS);
    
    py::class_<WaferVision::MeasurementResult>(m, "MeasurementResult")
        .def(py::init<>())
        .def_readwrite("type", &WaferVision::MeasurementResult::type)
        .def_readwrite("value", &WaferVision::MeasurementResult::value)
        .def_readwrite("uncertainty", &WaferVision::MeasurementResult::uncertainty)
        .def_readwrite("location", &WaferVision::MeasurementResult::location)
        .def_readwrite("unit", &WaferVision::MeasurementResult::unit)
        .def_readwrite("valid", &WaferVision::MeasurementResult::valid);
    
    py::class_<WaferVision::Metrology>(m, "Metrology")
        .def(py::init<>())
        .def("run_metrology", &WaferVision::Metrology::runMetrology)
        .def("measure_critical_dimension", &WaferVision::Metrology::measureCriticalDimension)
        .def("measure_line_width", &WaferVision::Metrology::measureLineWidth)
        .def("measure_space_width", &WaferVision::Metrology::measureSpaceWidth)
        .def("measure_overlay_error", &WaferVision::Metrology::measureOverlayError)
        .def("calculate_focus_metric", &WaferVision::Metrology::calculateFocusMetric)
        .def("set_calibration_factor", &WaferVision::Metrology::setCalibrationFactor)
        .def("set_measurement_tolerance", &WaferVision::Metrology::setMeasurementTolerance);
}

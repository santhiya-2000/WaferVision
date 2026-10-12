# WaferVision API Documentation

## C++ API

### Core Classes

#### ImageProcessor

Image processing and enhancement operations.

```cpp
#include "core/ImageProcessor.hpp"

WaferVision::ImageProcessor processor;
processor.setUseGpu(true);

cv::Mat image = processor.loadImage("wafer.png");
cv::Mat processed = processor.preprocessImage(image);
cv::Mat denoised = processor.denoiseImage(image);
cv::Mat enhanced = processor.enhanceContrast(image);
```

**Methods:**
- `loadImage(const std::string& path)` - Load image from file
- `preprocessImage(const cv::Mat& image)` - Full preprocessing pipeline
- `normalizeImage(const cv::Mat& image)` - Normalize pixel values
- `denoiseImage(const cv::Mat& image)` - Remove noise
- `enhanceContrast(const cv::Mat& image)` - Enhance contrast
- `sharpenImage(const cv::Mat& image)` - Sharpen image
- `adjustBrightness(const cv::Mat& image, double alpha, double beta)` - Adjust brightness
- `rotateImage(const cv::Mat& image, double angle)` - Rotate image
- `scaleImage(const cv::Mat& image, double scale)` - Scale image
- `translateImage(const cv::Mat& image, int dx, int dy)` - Translate image
- `isGpuAvailable()` - Check GPU availability
- `setUseGpu(bool useGpu)` - Enable/disable GPU acceleration

#### WaferData

Container for wafer image and associated data.

```cpp
#include "core/WaferData.hpp"

WaferVision::WaferData wafer("wafer.png");
wafer.setWaferDiameter(300.0);

cv::Mat image = wafer.getImage();
std::vector<Defect> defects = wafer.getDefects();
```

**Methods:**
- `loadImage(const std::string& path)` - Load wafer image
- `getImage()` - Get wafer image
- `getWaferMap()` - Get wafer map data
- `getMetrologyData()` - Get metrology measurements
- `setWaferDiameter(double diameter)` - Set wafer diameter
- `getWaferDiameter()` - Get wafer diameter
- `setWaferCenter(const cv::Point2f& center)` - Set wafer center
- `getWaferCenter()` - Get wafer center
- `addDefect(const Defect& defect)` - Add defect
- `clearDefects()` - Clear all defects
- `getDefects()` - Get all defects
- `getDefectCount()` - Get defect count
- `setMetadata(const std::string& key, const std::string& value)` - Set metadata
- `getMetadata(const std::string& key)` - Get metadata

#### InspectionMetrics

Performance and quality metrics tracking.

```cpp
#include "core/InspectionMetrics.hpp"

WaferVision::InspectionMetrics metrics;
metrics.recordMetric("defect_count", 42, "count");
metrics.recordTiming("processing", 1250.5);

std::string report = metrics.generateReport();
metrics.exportToJson("metrics.json");
```

**Methods:**
- `recordMetric(const std::string& name, double value, const std::string& unit)` - Record metric
- `recordTiming(const std::string& operation, double milliseconds)` - Record timing
- `getMetric(const std::string& name)` - Get metric value
- `getProcessingTime()` - Get total processing time
- `getThroughput()` - Get throughput (wafers/hour)
- `getDefectDetectionRate()` - Get defect detection rate
- `getFalsePositiveRate()` - Get false positive rate
- `generateReport()` - Generate text report
- `exportToCsv(const std::string& path)` - Export to CSV
- `exportToJson(const std::string& path)` - Export to JSON
- `reset()` - Reset all metrics

### Algorithm Classes

#### WaferAlignment

Wafer alignment and orientation detection.

```cpp
#include "algorithms/WaferAlignment.hpp"

WaferVision::WaferAlignment alignment;
alignment.setExpectedDiameter(300.0);

WaferVision::AlignmentResult result = alignment.alignWafer(image);
if (result.success) {
    cv::Mat aligned = alignment.applyAlignment(image, result);
}
```

**Methods:**
- `alignWafer(const cv::Mat& image, AlignmentMethod method)` - Align wafer
- `alignByCenterOfMass(const cv::Mat& image)` - Center of mass alignment
- `alignByHoughCircle(const cv::Mat& image)` - Hough circle alignment
- `alignByTemplateMatching(const cv::Mat& image, const cv::Mat& templateImage)` - Template matching
- `alignByEdgeDetection(const cv::Mat& image)` - Edge-based alignment
- `alignHybrid(const cv::Mat& image)` - Hybrid alignment
- `applyAlignment(const cv::Mat& image, const AlignmentResult& result)` - Apply alignment
- `detectFlat(const cv::Mat& image)` - Detect wafer flat
- `calculateFlatAngle(const cv::Mat& image, const cv::Point2f& center)` - Calculate flat angle
- `setExpectedDiameter(double diameter)` - Set expected diameter
- `setTolerance(double tolerance)` - Set alignment tolerance

#### DefectDetection

Defect detection with edge reduction and pattern processing.

```cpp
#include "algorithms/DefectDetection.hpp"

WaferVision::DefectDetection detector;
WaferVision::DetectionConfig config;
config.useEdgeReduction = true;
config.usePatternProcessing = true;
config.sensitivity = 0.5;

std::vector<Defect> defects = detector.detectDefects(image, config);
```

**Methods:**
- `detectDefects(const cv::Mat& image, const DetectionConfig& config)` - Detect defects
- `detectParticles(const cv::Mat& image)` - Detect particle defects
- `detectScratches(const cv::Mat& image)` - Detect scratch defects
- `detectPatternErrors(const cv::Mat& image, const cv::Mat& reference)` - Detect pattern errors
- `applyEdgeReduction(const cv::Mat& image)` - Apply edge reduction
- `applyPatternProcessing(const cv::Mat& image)` - Apply pattern processing
- `setConfig(const DetectionConfig& config)` - Set detection config
- `getConfig()` - Get detection config

#### Metrology

Metrology measurements and analysis.

```cpp
#include "algorithms/Metrology.hpp"

WaferVision::Metrology metrology;
metrology.setCalibrationFactor(1.0);

WaferVision::MetrologyData data = metrology.runMetrology(image, waferMap);
```

**Methods:**
- `runMetrology(const cv::Mat& image, const WaferMap& waferMap)` - Run full metrology
- `measureCriticalDimension(const cv::Mat& image, const cv::Point2f& location)` - Measure CD
- `measureLineWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end)` - Measure line width
- `measureSpaceWidth(const cv::Mat& image, const cv::Point2f& start, const cv::Point2f& end)` - Measure space width
- `measureOverlayError(const cv::Mat& currentLayer, const cv::Mat& referenceLayer)` - Measure overlay
- `calculateFocusMetric(const cv::Mat& image)` - Calculate focus metric
- `setCalibrationFactor(double factor)` - Set calibration factor
- `setMeasurementTolerance(double tolerance)` - Set measurement tolerance

## Python API

### Basic Usage

```python
import cv2
import numpy as np
from wafervision_python import WaferData, ImageProcessor, DefectDetection

# Load wafer
wafer = WaferData("wafer.png")

# Process image
processor = ImageProcessor()
processor.set_use_gpu(True)
processed = processor.preprocess_image(wafer.get_image())

# Detect defects
detector = DefectDetection()
config = DetectionConfig()
config.use_edge_reduction = True
config.sensitivity = 0.5

defects = detector.detect_defects(processed, config)
print(f"Found {len(defects)} defects")
```

### Demo Script

```python
python3 src/python/scripts/wafervision_demo.py
```

### Data Generation

```python
python3 src/python/scripts/generate_wafer_data.py --output data/simulated --count 10
```

## C# API

### Main Form

```csharp
using WaferVision.UI;

var form = new MainForm();
Application.Run(form);
```

### Building C# Components

```bash
cd src/csharp/WaferVision.UI
dotnet restore
dotnet build
dotnet run
```

# WaferVision Architecture

## System Overview

WaferVision is a multi-language semiconductor inspection system designed for high-throughput wafer analysis with GPU acceleration.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   C# UI      │  │  Python API  │  │  C++ CLI     │ │
│  │  (Windows)   │  │  (Scripting) │  │  (Console)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Processing Layer                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Image Processing Module                 │  │
│  │  - Preprocessing                                  │  │
│  │  - Enhancement                                   │  │
│  │  - Transformation                                │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Wafer Data Management                   │  │
│  │  - Image storage                                 │  │
│  │  - Defect tracking                               │  │
│  │  - Metadata management                           │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Metrics Collection                      │  │
│  │  - Performance tracking                           │  │
│  │  - Quality metrics                                │  │
│  │  - Report generation                              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Algorithm Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Alignment  │  │  Defect Det. │  │  Metrology    │ │
│  │              │  │              │  │              │ │
│  │ - CoM        │  │ - Particles  │  │ - CD         │ │
│  │ - Hough      │  │ - Scratches  │  │ - Overlay    │ │
│  │ - Template   │  │ - Patterns   │  │ - Focus      │ │
│  │ - Edge       │  │ - Edge Red.  │  │ - Thickness  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Acceleration Layer                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │              GPU Acceleration                      │  │
│  │  - CUDA kernels                                   │  │
│  │  - Image processing                              │  │
│  │  - Defect detection                              │  │
│  │  - Automatic CPU fallback                         │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │              CPU Processing                       │  │
│  │  - OpenCV operations                             │  │
│  │  - Fallback path                                 │  │
│  │  - Single-threaded                               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   OpenCV     │  │    CUDA      │  │  File I/O    │ │
│  │              │  │              │  │              │ │
│  │ - imgproc    │  │ - Runtime    │  │ - Images     │ │
│  │ - features2d │  │ - Kernels    │  │ - JSON       │ │
│  │ - photo      │  │ - Memory     │  │ - CSV        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Image Processing Module

**Responsibilities:**
- Image loading and preprocessing
- Noise reduction and enhancement
- Geometric transformations
- GPU/CPU processing selection

**Key Classes:**
- `ImageProcessor` - Main processing interface
- GPU kernels in `GpuImageProcessor.cu`

### 2. Wafer Data Management

**Responsibilities:**
- Wafer image storage
- Defect data tracking
- Metrology data storage
- Metadata management

**Key Classes:**
- `WaferData` - Main data container
- `WaferMap` - Wafer geometry and defects
- `MetrologyData` - Measurement results

### 3. Algorithm Layer

#### Wafer Alignment
- Center of Mass (CoM) alignment
- Hough Circle detection
- Template matching
- Edge-based alignment
- Hybrid approach combining multiple methods

#### Defect Detection
- Particle detection using adaptive thresholding
- Scratch detection using line detection
- Pattern error detection using reference comparison
- Edge reduction (reduces false positives by ~40%)
- Pattern processing for periodic structures

#### Metrology
- Critical Dimension (CD) measurement
- Line/space width measurement
- Overlay error measurement
- Focus metric calculation
- Thickness estimation
- Surface roughness measurement

### 4. GPU Acceleration

**CUDA Kernels:**
- `denoiseKernel` - Image denoising
- `enhanceContrastKernel` - Contrast enhancement
- `sobelKernel` - Edge detection
- `thresholdDefectKernel` - Threshold-based defect detection
- `localContrastKernel` - Local contrast defect detection
- `erosionKernel` / `dilationKernel` - Morphological operations

**Performance:**
- ~3× faster processing compared to CPU-only
- Automatic CPU fallback when GPU unavailable
- Memory-efficient for large images (up to 16K×16K)

### 5. Metrics Collection

**Tracked Metrics:**
- Processing time per operation
- Throughput (wafers/hour)
- Defect detection rate
- False positive rate
- Custom measurement metrics

**Export Formats:**
- Text reports
- CSV files
- JSON files

## Data Flow

### Inspection Workflow

```
1. Load Wafer Image
   ↓
2. Preprocess (Denoise → Normalize → Enhance)
   ↓
3. Align Wafer (Center → Rotate)
   ↓
4. Detect Defects (Edge Reduction → Pattern Processing)
   ↓
5. Run Metrology (CD → Overlay → Focus)
   ↓
6. Collect Metrics
   ↓
7. Generate Reports
```

### GPU Processing Flow

```
Image Input
   ↓
GPU Available?
   ├─ Yes → Upload to GPU
   │        ↓
   │     CUDA Kernel Execution
   │        ↓
   │     Download to CPU
   │
   └─ No → CPU Processing (OpenCV)
           ↓
        Result Output
```

## Technology Stack

### Core Languages
- **C++17** - Core processing engine
- **CUDA** - GPU acceleration
- **C#** - Windows UI (optional)
- **Python** - Scripting interface (optional)

### Libraries
- **OpenCV 4.5+** - Computer vision operations
- **CUDA Toolkit 11.0+** - GPU computing
- **pybind11** - Python bindings
- **.NET 6.0+** - C# runtime

### Build System
- **CMake 3.20+** - Cross-platform build configuration
- **dotnet** - C# project building

## Performance Characteristics

### Processing Speed
- GPU: ~3× faster than CPU
- CPU fallback: Full functionality with reduced speed
- Typical wafer (1024×1024): <500ms with GPU

### Memory Usage
- Base: ~50MB for 1024×1024 image
- GPU: Additional ~100MB for GPU buffers
- Large wafers (16K×16K): ~12GB RAM required

### Accuracy
- Edge reduction: ~40% reduction in false positives
- Alignment accuracy: <1 pixel error
- Metrology accuracy: Sub-pixel precision

## Extension Points

### Adding New Algorithms
1. Create new class in `src/cpp/algorithms/`
2. Inherit from appropriate base class
3. Implement required methods
4. Add to CMakeLists.txt
5. Create Python bindings if needed

### Adding New GPU Kernels
1. Add kernel to `src/cpp/cuda/`
2. Create wrapper function with `extern "C"`
3. Update CPU fallback in main class
4. Add to CMakeLists.txt

### Adding New Metrics
1. Add metric type to `InspectionMetrics`
2. Implement recording method
3. Add to report generation
4. Update export functions

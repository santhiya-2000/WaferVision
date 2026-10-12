# WaferVision - Semiconductor Wafer Inspection & Metrology System

A comprehensive semiconductor inspection application implementing image-processing, wafer alignment, defect detection, and metrology workflows with GPU acceleration.

## Features

- **GPU-Accelerated Processing**: CUDA-based image processing with CPU fallback, achieving ~3× faster processing
- **Wafer Alignment**: Advanced algorithms for precise wafer positioning and orientation
- **Defect Detection**: Edge-reduction and pattern-processing techniques reducing false detections by ~40%
- **Metrology Workflows**: Comprehensive measurement and analysis tools
- **Multi-Language Support**: C++17 core, C# visualization, Python scripting interface
- **Visualization**: Real-time inspection metrics and equipment-oriented workflows

## Technology Stack

- **C++17**: Core image processing and algorithms
- **CUDA**: GPU acceleration with CPU fallback
- **C#**: Visualization and UI layer
- **Python**: Scripting interface and data analysis
- **OpenCV**: Computer vision and image processing

## Project Structure

```
WaferVision/
├── src/
│   ├── cpp/           # C++ core implementation
│   ├── csharp/        # C# visualization layer
│   └── python/        # Python bindings and scripts
├── data/
│   ├── simulated/     # Simulated wafer data
│   └── public/        # Public datasets
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── build/             # Build artifacts
```

## Building

### Prerequisites

- CMake 3.20+
- C++17 compatible compiler
- CUDA Toolkit 11.0+
- OpenCV 4.5+
- Python 3.8+ (for Python bindings)
- .NET 6.0+ (for C# components)

### Build Steps

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build .

# Install (optional)
cmake --install .
```

### Building Python Bindings

```bash
pip install pybind11
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
cmake --build .
```

### Building C# Components

```bash
cd src/csharp
dotnet restore
dotnet build
```

## Usage

### C++ CLI

```bash
./build/wafervision_cli --input data/simulated/wafer_001.png --output results/
```

### Python Interface

```python
import wafervision

# Load wafer image
wafer = wafervision.WaferData("data/simulated/wafer_001.png")

# Align wafer
aligned = wafervision.align_wafer(wafer)

# Detect defects
defects = wafervision.detect_defects(aligned)

# Run metrology
metrics = wafervision.run_metrology(aligned)
```

### C# Visualization

```bash
cd src/csharp
dotnet run --project WaferVision.UI
```

## Data

### Simulated Data

The project includes simulated wafer data generated using realistic semiconductor patterns:
- Random defect injection
- Various defect types (particles, scratches, pattern errors)
- Different wafer sizes and resolutions

### Public Data

Compatible with public semiconductor datasets:
- SEM image datasets
- Wafer map datasets
- Defect classification datasets

## Performance

- **GPU Acceleration**: ~3× faster processing compared to CPU-only
- **False Detection Reduction**: ~40% improvement through edge-reduction
- **Memory Efficiency**: Optimized for large wafer images (up to 16K×16K)

## License

Proprietary - Contact for licensing information

## Contributing

Internal project - See contribution guidelines for team members.

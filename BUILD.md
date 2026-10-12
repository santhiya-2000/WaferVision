# Building WaferVision

## Prerequisites

### Required
- CMake 3.20 or higher
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CUDA Toolkit 11.0 or higher (for GPU acceleration)
- OpenCV 4.5 or higher

### Optional (for Python bindings)
- Python 3.8 or higher
- pybind11 2.9 or higher

### Optional (for C# UI)
- .NET 6.0 SDK or higher
- Windows OS (C# UI is Windows-only)

## Installing Dependencies

### macOS (Homebrew)
```bash
brew install cmake opencv
# Install CUDA from NVIDIA website
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install cmake build-essential libopencv-dev
# Install CUDA from NVIDIA repository
```

### Windows
- Install CMake from https://cmake.org/
- Install Visual Studio 2019 or later with C++ support
- Install CUDA Toolkit from NVIDIA
- Install OpenCV (prebuilt binaries or build from source)

## Building the C++ Core

### Step 1: Create Build Directory
```bash
cd WaferVision
mkdir build
cd build
```

### Step 2: Configure with CMake
```bash
cmake ..
```

### Step 3: Build
```bash
cmake --build . --config Release
```

### Step 4: Install (Optional)
```bash
cmake --install .
```

## Building with Python Bindings

### Step 1: Install pybind11
```bash
pip install pybind11
```

### Step 2: Configure with Python
```bash
cmake -DPYTHON_EXECUTABLE=$(which python) ..
```

### Step 3: Build
```bash
cmake --build . --config Release
```

### Step 4: Install Python Module
```bash
pip install .
```

## Building C# Components

### Step 1: Restore Dependencies
```bash
cd src/csharp/WaferVision.UI
dotnet restore
```

### Step 2: Build
```bash
dotnet build --configuration Release
```

### Step 3: Run
```bash
dotnet run
```

## Building Options

### Disable CUDA (CPU-only)
```bash
cmake -DENABLE_CUDA=OFF ..
```

### Enable Debug Symbols
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Custom Installation Prefix
```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

## Running the Application

### C++ CLI
```bash
./build/wafervision_cli --input data/simulated/wafer_000.png \
                        --output results/ \
                        --align \
                        --detect \
                        --metrology
```

### Python Demo
```bash
python3 src/python/scripts/wafervision_demo.py
```

### Generate Simulated Data
```bash
python3 src/python/scripts/generate_wafer_data.py --output data/simulated --count 10
```

## Troubleshooting

### CUDA Not Found
- Ensure CUDA Toolkit is installed
- Set `CUDA_PATH` environment variable
- Try `-DENABLE_CUDA=OFF` for CPU-only build

### OpenCV Not Found
- Install OpenCV development files
- Set `OpenCV_DIR` to OpenCV installation
- Use package manager (apt, brew, vcpkg)

### Python Bindings Fail
- Ensure pybind11 is installed
- Check Python version compatibility
- Verify CMake can find Python

### C# Build Fails
- Ensure .NET 6.0 SDK is installed
- Run `dotnet --version` to verify
- Restore NuGet packages with `dotnet restore`

## Performance Optimization

### GPU Acceleration
- Ensure CUDA drivers are up to date
- Use appropriate GPU device ID in config
- Monitor GPU memory usage for large images

### Compilation Flags
- Release builds are significantly faster
- Enable compiler optimizations (`-O3` for GCC/Clang)
- Use native architecture flags (`-march=native`)

### Memory Management
- For large wafers (>4K), increase system RAM
- GPU memory should be >4GB for best performance
- Consider batch processing for multiple wafers

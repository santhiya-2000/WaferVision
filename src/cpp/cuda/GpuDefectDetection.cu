#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace WaferVision {

// CUDA kernel for threshold-based defect detection
__global__ void thresholdDefectKernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = (input[idx] < threshold) ? 255 : 0;
}

// CUDA kernel for local contrast defect detection
__global__ void localContrastKernel(const unsigned char* input, unsigned char* output,
                                    int width, int height, int windowSize, float contrastThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfWindow = windowSize / 2;
    float localMean = 0.0f;
    int count = 0;
    
    // Calculate local mean
    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                localMean += input[ny * width + nx];
                count++;
            }
        }
    }
    localMean /= count;
    
    // Calculate local standard deviation
    float localStdDev = 0.0f;
    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float diff = input[ny * width + nx] - localMean;
                localStdDev += diff * diff;
            }
        }
    }
    localStdDev = sqrtf(localStdDev / count);
    
    // Mark as defect if contrast exceeds threshold
    int idx = y * width + x;
    float contrast = fabsf(static_cast<float>(input[idx]) - localMean) / (localStdDev + 1e-6f);
    output[idx] = (contrast > contrastThreshold) ? 255 : 0;
}

// CUDA kernel for morphological operations (erosion)
__global__ void erosionKernel(const unsigned char* input, unsigned char* output,
                             int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfKernel = kernelSize / 2;
    unsigned char minVal = 255;
    
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                minVal = fminf(minVal, input[ny * width + nx]);
            }
        }
    }
    
    output[y * width + x] = minVal;
}

// CUDA kernel for morphological operations (dilation)
__global__ void dilationKernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfKernel = kernelSize / 2;
    unsigned char maxVal = 0;
    
    for (int dy = -halfKernel; dy <= halfKernel; dy++) {
        for (int dx = -halfKernel; dx <= halfKernel; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                maxVal = fmaxf(maxVal, input[ny * width + nx]);
            }
        }
    }
    
    output[y * width + x] = maxVal;
}

// Wrapper functions
extern "C" {

cv::Mat gpuThresholdDefectWrapper(const cv::Mat& input, unsigned char threshold) {
    cv::Mat output(input.size(), CV_8U);
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    thresholdDefectKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, threshold);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

cv::Mat gpuLocalContrastWrapper(const cv::Mat& input, int windowSize, float contrastThreshold) {
    cv::Mat output(input.size(), CV_8U);
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    localContrastKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, 
                                                  windowSize, contrastThreshold);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

cv::Mat gpuErosionWrapper(const cv::Mat& input, int kernelSize) {
    cv::Mat output(input.size(), CV_8U);
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    erosionKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, kernelSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

cv::Mat gpuDilationWrapper(const cv::Mat& input, int kernelSize) {
    cv::Mat output(input.size(), CV_8U);
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    dilationKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, kernelSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

} // extern "C"

} // namespace WaferVision

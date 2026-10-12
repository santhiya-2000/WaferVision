#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace WaferVision {

// CUDA kernel for image denoising
__global__ void denoiseKernel(const unsigned char* input, unsigned char* output, 
                              int width, int height, int windowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfWindow = windowSize / 2;
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    
    output[y * width + x] = static_cast<unsigned char>(sum / count);
}

// CUDA kernel for contrast enhancement
__global__ void enhanceContrastKernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float value = static_cast<float>(input[idx]);
    value = alpha * value + beta;
    output[idx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)));
}

// CUDA kernel for Sobel edge detection
__global__ void sobelKernel(const unsigned char* input, unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }
    
    // Sobel kernels
    int gx = -input[(y-1) * width + (x-1)] + input[(y-1) * width + (x+1)]
             - 2 * input[y * width + (x-1)] + 2 * input[y * width + (x+1)]
             - input[(y+1) * width + (x-1)] + input[(y+1) * width + (x+1)];
             
    int gy = -input[(y-1) * width + (x-1)] - 2 * input[(y-1) * width + x] - input[(y-1) * width + (x+1)]
             + input[(y+1) * width + (x-1)] + 2 * input[(y+1) * width + x] + input[(y+1) * width + (x+1)];
    
    int magnitude = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
    output[y * width + x] = static_cast<unsigned char>(fminf(255, magnitude));
}

// Wrapper functions for CUDA operations
extern "C" {

cv::Mat gpuDenoiseWrapper(const cv::Mat& input, int windowSize = 3) {
    cv::Mat output(input.size(), input.type());
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    denoiseKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, windowSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

cv::Mat gpuEnhanceContrastWrapper(const cv::Mat& input, float alpha = 1.5f, float beta = 0.0f) {
    cv::Mat output(input.size(), input.type());
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    enhanceContrastKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows, alpha, beta);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

cv::Mat gpuSobelWrapper(const cv::Mat& input) {
    cv::Mat output(input.size(), input.type());
    
    unsigned char* d_input;
    unsigned char* d_output;
    size_t size = input.rows * input.cols * sizeof(unsigned char);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);
    
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, input.cols, input.rows);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

} // extern "C"

} // namespace WaferVision

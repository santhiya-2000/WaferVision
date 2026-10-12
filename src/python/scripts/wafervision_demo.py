#!/usr/bin/env python3
"""
WaferVision Python Demo Script
Demonstrates the Python API for wafer inspection and metrology
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# This would import the compiled Python bindings
# import wafervision_python as wv

class WaferVisionDemo:
    """Demo class for WaferVision Python API"""
    
    def __init__(self):
        self.image = None
        self.defects = []
        self.metrics = {}
    
    def load_image(self, path):
        """Load a wafer image"""
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Failed to load image: {path}")
        print(f"Loaded image: {path}")
        print(f"Image size: {self.image.shape}")
        return self.image
    
    def preprocess_image(self):
        """Preprocess the wafer image"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(self.image, None, 10, 7, 21)
        
        # Normalize
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(normalized)
        
        self.image = enhanced
        print("Preprocessing completed")
        return self.image
    
    def align_wafer(self):
        """Align the wafer using center of mass"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Threshold
        _, binary = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find center using moments
        m = cv2.moments(binary)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        
        # Calculate translation
        center_x, center_y = self.image.shape[1] // 2, self.image.shape[0] // 2
        dx = center_x - cx
        dy = center_y - cy
        
        # Apply translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
        
        self.image = aligned
        print(f"Wafer aligned. Translation: ({dx}, {dy})")
        return self.image
    
    def detect_defects(self, sensitivity=0.5):
        """Detect defects in the wafer image"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Edge detection
        edges = cv2.Canny(self.image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.defects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 10 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                self.defects.append({
                    'id': i,
                    'position': (x + w//2, y + h//2),
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'type': 'unknown'
                })
        
        print(f"Detected {len(self.defects)} defects")
        return self.defects
    
    def run_metrology(self):
        """Run metrology measurements"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate contrast
        contrast = self.image.std()
        
        self.metrics = {
            'sharpness': sharpness,
            'contrast': contrast,
            'mean_intensity': self.image.mean(),
            'defect_count': len(self.defects)
        }
        
        print("Metrology results:")
        for key, value in self.metrics.items():
            print(f"  {key}: {value:.2f}")
        
        return self.metrics
    
    def visualize_results(self, output_path=None):
        """Visualize inspection results"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Processed Image')
        axes[0].axis('off')
        
        # Defect map
        defect_map = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for defect in self.defects:
            x, y, w, h = defect['bounding_box']
            cv2.rectangle(defect_map, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        axes[1].imshow(cv2.cvtColor(defect_map, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Defect Map ({len(self.defects)} defects)')
        axes[1].axis('off')
        
        # Metrics
        if self.metrics:
            metric_names = list(self.metrics.keys())
            metric_values = list(self.metrics.values())
            axes[2].bar(metric_names, metric_values)
            axes[2].set_title('Metrology Metrics')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path):
        """Generate inspection report"""
        with open(output_path, 'w') as f:
            f.write("WaferVision Inspection Report\n")
            f.write("=" * 40 + "\n\n")
            
            if self.image is not None:
                f.write(f"Image Size: {self.image.shape}\n")
            
            f.write(f"Defect Count: {len(self.defects)}\n\n")
            
            f.write("Defect Details:\n")
            for defect in self.defects:
                f.write(f"  ID: {defect['id']}, Position: {defect['position']}, "
                       f"Area: {defect['area']:.2f}\n")
            
            f.write("\nMetrology Metrics:\n")
            for key, value in self.metrics.items():
                f.write(f"  {key}: {value:.2f}\n")
        
        print(f"Report saved to {output_path}")


def main():
    """Main demo function"""
    print("WaferVision Python Demo")
    print("=" * 40)
    
    demo = WaferVisionDemo()
    
    # Example usage with simulated data
    try:
        # Generate simulated wafer image
        print("\nGenerating simulated wafer image...")
        demo.image = generate_simulated_wafer(512, 512)
        
        # Process the wafer
        print("\nProcessing wafer...")
        demo.preprocess_image()
        demo.align_wafer()
        demo.detect_defects()
        demo.run_metrology()
        
        # Visualize results
        print("\nGenerating visualization...")
        demo.visualize_results('inspection_results.png')
        
        # Generate report
        print("\nGenerating report...")
        demo.generate_report('inspection_report.txt')
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


def generate_simulated_wafer(width, height, defect_count=20):
    """Generate a simulated wafer image with defects"""
    # Create circular wafer
    image = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 20
    cv2.circle(image, center, radius, 200, -1)
    
    # Add some texture
    noise = np.random.normal(0, 10, (height, width)).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Add flat
    flat_start = (center[0] - radius, center[1] - 10)
    flat_end = (center[0] - radius + 30, center[1] + 10)
    cv2.rectangle(image, flat_start, flat_end, 0, -1)
    
    # Add defects
    for _ in range(defect_count):
        # Random position within wafer
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius - 20)
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        
        # Random defect type
        defect_type = np.random.choice(['particle', 'scratch'])
        
        if defect_type == 'particle':
            size = np.random.randint(3, 15)
            cv2.circle(image, (x, y), size, 50, -1)
        else:  # scratch
            length = np.random.randint(10, 30)
            angle = np.random.uniform(0, np.pi)
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            cv2.line(image, (x, y), (end_x, end_y), 50, 2)
    
    return image


if __name__ == "__main__":
    main()

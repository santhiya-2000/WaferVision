#!/usr/bin/env python3
"""
Simple WaferVision demo without matplotlib dependency
"""

import cv2
import numpy as np
import json
from pathlib import Path

def load_wafer_data(image_path):
    """Load wafer image and defect data"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    defect_path = image_path.replace('.png', '_defects.json')
    
    defects = []
    if Path(defect_path).exists():
        with open(defect_path, 'r') as f:
            defects = json.load(f)
    
    return image, defects

def process_wafer(image):
    """Simple wafer processing"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    # Normalize
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    
    # Enhance contrast
    enhanced = cv2.equalizeHist(normalized)
    
    return enhanced

def detect_defects_simple(image):
    """Simple defect detection"""
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 10 < area < 1000:
            x, y, w, h = cv2.boundingRect(contour)
            detected.append({
                'id': i,
                'position': (x + w//2, y + h//2),
                'bounding_box': (x, y, w, h),
                'area': area
            })
    
    return detected

def save_defect_map(image, defects, output_path):
    """Save defect map with bounding boxes"""
    defect_map = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for defect in defects:
        x, y, w, h = defect['bounding_box']
        cv2.rectangle(defect_map, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imwrite(output_path, defect_map)

def main():
    print("WaferVision Simple Demo")
    print("=" * 40)
    
    # Use existing simulated data
    data_dir = Path("data/simulated")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Process first wafer
    wafer_files = sorted(data_dir.glob("wafer_*.png"))
    
    if not wafer_files:
        print("No wafer files found")
        return
    
    wafer_path = wafer_files[0]
    print(f"\nProcessing: {wafer_path.name}")
    
    # Load data
    image, ground_truth = load_wafer_data(str(wafer_path))
    print(f"Image size: {image.shape}")
    print(f"Ground truth defects: {len(ground_truth)}")
    
    # Process
    print("\nProcessing wafer...")
    processed = process_wafer(image)
    
    # Detect defects
    print("Detecting defects...")
    detected = detect_defects_simple(processed)
    print(f"Detected defects: {len(detected)}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "processed.png"), processed)
    save_defect_map(processed, detected, str(output_dir / "defect_map.png"))
    
    print(f"\nResults saved to {output_dir}/")
    print("  - processed.png")
    print("  - defect_map.png")
    
    # Print defect details
    if detected:
        print("\nDetected defects:")
        for defect in detected[:5]:  # Show first 5
            print(f"  ID {defect['id']}: Position {defect['position']}, Area {defect['area']:.1f}")
        if len(detected) > 5:
            print(f"  ... and {len(detected) - 5} more")

if __name__ == "__main__":
    main()

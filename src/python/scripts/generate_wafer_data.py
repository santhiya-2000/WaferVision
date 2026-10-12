#!/usr/bin/env python3
"""
Generate simulated wafer data for testing and demonstration
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict


class WaferGenerator:
    """Generate realistic simulated wafer images"""
    
    def __init__(self, size: Tuple[int, int] = (1024, 1024), 
                 diameter_mm: float = 300.0):
        self.size = size
        self.diameter_mm = diameter_mm
        self.pixels_per_mm = size[0] / diameter_mm
    
    def generate_wafer(self, defect_count: int = 50, 
                      pattern_type: str = 'grid') -> np.ndarray:
        """Generate a wafer image with defects"""
        image = np.zeros(self.size, dtype=np.uint8)
        
        # Create wafer circle
        center = (self.size[0] // 2, self.size[1] // 2)
        radius = min(self.size) // 2 - 20
        cv2.circle(image, center, radius, 180, -1)
        
        # Add pattern
        if pattern_type == 'grid':
            image = self._add_grid_pattern(image, center, radius)
        elif pattern_type == 'radial':
            image = self._add_radial_pattern(image, center, radius)
        
        # Add texture
        image = self._add_texture(image)
        
        # Add flat
        self._add_flat(image, center, radius)
        
        # Add defects
        defects = self._add_defects(image, center, radius, defect_count)
        
        return image, defects
    
    def _add_grid_pattern(self, image: np.ndarray, center: Tuple[int, int], 
                         radius: int) -> np.ndarray:
        """Add grid pattern to wafer"""
        spacing = 20
        for x in range(center[0] - radius, center[0] + radius, spacing):
            if 0 <= x < image.shape[1]:
                cv2.line(image, (x, center[1] - radius), 
                        (x, center[1] + radius), 150, 1)
        
        for y in range(center[1] - radius, center[1] + radius, spacing):
            if 0 <= y < image.shape[0]:
                cv2.line(image, (center[0] - radius, y), 
                        (center[0] + radius, y), 150, 1)
        
        return image
    
    def _add_radial_pattern(self, image: np.ndarray, center: Tuple[int, int], 
                           radius: int) -> np.ndarray:
        """Add radial pattern to wafer"""
        for angle in np.linspace(0, 2*np.pi, 36):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            cv2.line(image, center, (x, y), 150, 1)
        
        return image
    
    def _add_texture(self, image: np.ndarray) -> np.ndarray:
        """Add realistic texture"""
        noise = np.random.normal(0, 5, self.size).astype(np.int16)
        image = cv2.add(image.astype(np.int16), noise).astype(np.uint8)
        return image
    
    def _add_flat(self, image: np.ndarray, center: Tuple[int, int], 
                 radius: int):
        """Add wafer flat"""
        flat_width = 30
        flat_height = 20
        x = center[0] - radius
        y = center[1] - flat_height // 2
        cv2.rectangle(image, (x, y), (x + flat_width, y + flat_height), 0, -1)
    
    def _add_defects(self, image: np.ndarray, center: Tuple[int, int], 
                    radius: int, count: int) -> List[Dict]:
        """Add various defect types"""
        defects = []
        defect_types = ['particle', 'scratch', 'pattern_error', 'missing_feature']
        
        for i in range(count):
            # Random position within wafer
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius - 30)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            
            defect_type = np.random.choice(defect_types)
            
            if defect_type == 'particle':
                size = np.random.randint(3, 20)
                severity = size / 20.0
                cv2.circle(image, (x, y), size, 50, -1)
                defects.append({
                    'id': i,
                    'type': 'particle',
                    'position': (x, y),
                    'size': size,
                    'severity': severity
                })
            
            elif defect_type == 'scratch':
                length = np.random.randint(10, 50)
                angle = np.random.uniform(0, np.pi)
                end_x = int(x + length * np.cos(angle))
                end_y = int(y + length * np.sin(angle))
                severity = length / 50.0
                cv2.line(image, (x, y), (end_x, end_y), 50, 2)
                defects.append({
                    'id': i,
                    'type': 'scratch',
                    'position': (x, y),
                    'length': length,
                    'severity': severity
                })
            
            elif defect_type == 'pattern_error':
                size = np.random.randint(10, 30)
                severity = size / 30.0
                cv2.rectangle(image, (x, y), (x + size, y + size), 50, -1)
                defects.append({
                    'id': i,
                    'type': 'pattern_error',
                    'position': (x, y),
                    'size': size,
                    'severity': severity
                })
            
            elif defect_type == 'missing_feature':
                size = np.random.randint(5, 15)
                severity = size / 15.0
                cv2.circle(image, (x, y), size, 255, -1)
                defects.append({
                    'id': i,
                    'type': 'missing_feature',
                    'position': (x, y),
                    'size': size,
                    'severity': severity
                })
        
        return defects
    
    def save_wafer(self, image: np.ndarray, defects: List[Dict], 
                  output_dir: Path, name: str):
        """Save wafer image and defect data"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_path = output_dir / f"{name}.png"
        cv2.imwrite(str(image_path), image)
        
        # Save defect data
        defect_path = output_dir / f"{name}_defects.json"
        with open(defect_path, 'w') as f:
            json.dump(defects, f, indent=2)
        
        print(f"Saved wafer to {image_path}")
        print(f"Saved defects to {defect_path}")
        
        return image_path, defect_path


def generate_dataset(output_dir: Path, num_wafers: int = 10,
                    size: Tuple[int, int] = (1024, 1024)):
    """Generate a dataset of simulated wafers"""
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = WaferGenerator(size)
    
    for i in range(num_wafers):
        print(f"Generating wafer {i+1}/{num_wafers}...")
        
        # Random defect count
        defect_count = np.random.randint(20, 100)
        
        # Random pattern type
        pattern_type = np.random.choice(['grid', 'radial', 'none'])
        
        # Generate wafer
        if pattern_type == 'none':
            image = np.zeros(size, dtype=np.uint8)
            center = (size[0] // 2, size[1] // 2)
            radius = min(size) // 2 - 20
            cv2.circle(image, center, radius, 180, -1)
            image = generator._add_texture(image)
            generator._add_flat(image, center, radius)
            defects = generator._add_defects(image, center, radius, defect_count)
        else:
            image, defects = generator.generate_wafer(defect_count, pattern_type)
        
        # Save
        generator.save_wafer(image, defects, output_dir, f"wafer_{i:03d}")
    
    print(f"\nDataset generation complete. {num_wafers} wafers saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate simulated wafer data')
    parser.add_argument('--output', type=str, default='data/simulated',
                       help='Output directory for generated data')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of wafers to generate')
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024],
                       help='Wafer image size (width height)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    size = tuple(args.size)
    
    generate_dataset(output_dir, args.count, size)

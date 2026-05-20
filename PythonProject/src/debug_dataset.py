#!/usr/bin/env python3
"""
Script debug để kiểm tra dataset loading
"""

import os
import numpy as np
from fire_feature_extractor import DatasetLoader

def debug_dataset():
    """Debug dataset loading"""
    print("🔍 Debugging dataset loading...")
    
    dataset_path = '../dataset'
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    
    print(f"📁 Train images dir: {train_images_dir}")
    print(f"📁 Exists: {os.path.exists(train_images_dir)}")
    
    if os.path.exists(train_images_dir):
        subdirs = [d for d in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, d))]
        print(f"📁 Subdirs: {subdirs}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(train_images_dir, subdir)
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"📁 {subdir}: {len(images)} images")
            
            # Kiểm tra label logic
            label = 1 if subdir.lower() == 'fire' else 0
            print(f"   Label: {label} ({'fire' if label == 1 else 'no_fire'})")
    
    # Test DatasetLoader
    print("\n🔍 Testing DatasetLoader...")
    loader = DatasetLoader(dataset_path)
    X, y, paths = loader.load_dataset(max_samples=10)
    
    print(f"📊 X shape: {X.shape}")
    print(f"📊 y shape: {y.shape}")
    print(f"📊 Unique labels: {set(y)}")
    print(f"📊 Label counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Kiểm tra paths
    print(f"📊 Sample paths:")
    for i, (path, label) in enumerate(zip(paths[:5], y[:5])):
        print(f"   {i+1}. {path} -> {label}")

if __name__ == "__main__":
    debug_dataset() 
#!/usr/bin/env python3
"""
Update cooccur_voc2007.json with percentile-based thresholds.
Adds rare_p10, rare_p25, rare_classes_p10, rare_classes_p25 fields.
"""

import os
import json
import numpy as np

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def main():
    script_dir = os.path.dirname(__file__)
    cooccur_path = os.path.join(script_dir, '..', 'cooccur_voc2007.json')
    
    if not os.path.exists(cooccur_path):
        print(f"Error: {cooccur_path} not found!")
        return
    
    with open(cooccur_path, 'r') as f:
        data = json.load(f)
    
    freq = np.array(data['freq'])
    freq_nonzero = freq[freq > 0]
    
    # Compute percentile thresholds
    rare_p10 = float(np.percentile(freq_nonzero, 10))
    rare_p25 = float(np.percentile(freq_nonzero, 25))
    
    # Identify rare classes
    rare_classes_p10 = [int(i) for i in range(len(freq)) if freq[i] > 0 and freq[i] < rare_p10]
    rare_classes_p25 = [int(i) for i in range(len(freq)) if freq[i] > 0 and freq[i] < rare_p25]
    
    print("="*60)
    print("VOC2007 Class Frequency Statistics")
    print("="*60)
    print(f"Total classes: {len(freq)}")
    print(f"Frequency range: {freq.min()} - {freq.max()}")
    print(f"Mean: {freq.mean():.1f}, Median: {np.median(freq):.1f}")
    print(f"\nPercentile Thresholds:")
    print(f"  P10 (极稀有): {rare_p10:.0f}")
    print(f"  P25 (稀有):   {rare_p25:.0f}")
    
    print(f"\n极稀有类 (freq < P10={rare_p10:.0f}):")
    for idx in rare_classes_p10:
        print(f"  [{idx:2d}] {VOC_CLASSES[idx]}: {freq[idx]}")
    
    print(f"\n稀有类 (P10 <= freq < P25={rare_p25:.0f}):")
    for idx in rare_classes_p25:
        if idx not in rare_classes_p10:
            print(f"  [{idx:2d}] {VOC_CLASSES[idx]}: {freq[idx]}")
    
    print(f"\n高频类 Top-5:")
    top_indices = np.argsort(freq)[::-1][:5]
    for idx in top_indices:
        print(f"  [{idx:2d}] {VOC_CLASSES[idx]}: {freq[idx]}")
    
    # Update data
    data['dataset'] = 'VOC2007'
    data['num_classes'] = 20
    data['class_names'] = VOC_CLASSES
    data['rare_p10'] = rare_p10
    data['rare_p25'] = rare_p25
    data['rare_classes_p10'] = rare_classes_p10
    data['rare_classes_p25'] = rare_classes_p25
    
    # Save updated file
    with open(cooccur_path, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Updated: {cooccur_path}")
    print("="*60)

if __name__ == '__main__':
    main()

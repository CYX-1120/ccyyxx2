#!/usr/bin/env python3
"""
Compute COCO2014 class frequency and co-occurrence statistics.
Generate cooccur_coco.json for Rare-Class Aware Mixing (分档版本).
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# COCO category ID to index mapping (COCO uses non-contiguous IDs)
COCO_CAT_ID_TO_IDX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def compute_coco_stats(annotation_file):
    print(f"Loading annotations from {annotation_file}...")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    num_classes = 80
    freq = np.zeros(num_classes, dtype=np.int64)
    cooccur = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Build image_id -> category indices mapping
    image_to_cats = defaultdict(set)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        cat_id = ann['category_id']
        if cat_id in COCO_CAT_ID_TO_IDX:
            cat_idx = COCO_CAT_ID_TO_IDX[cat_id]
            image_to_cats[image_id].add(cat_idx)
    
    print(f"Processing {len(image_to_cats)} images...")
    
    # Compute frequency and co-occurrence
    for image_id, cat_indices in image_to_cats.items():
        cat_list = list(cat_indices)
        for cat_idx in cat_list:
            freq[cat_idx] += 1
        # Co-occurrence (symmetric)
        for i in range(len(cat_list)):
            for j in range(i+1, len(cat_list)):
                cooccur[cat_list[i], cat_list[j]] += 1
                cooccur[cat_list[j], cat_list[i]] += 1
    
    return freq, cooccur

def main():
    coco_root = '/export/data/MSCOCO2014'
    train_ann_file = os.path.join(coco_root, 'annotations', 'instances_train2014.json')
    
    if not os.path.exists(train_ann_file):
        print(f"Error: Annotation file not found: {train_ann_file}")
        sys.exit(1)
    
    freq, cooccur = compute_coco_stats(train_ann_file)
    
    # Compute percentile thresholds
    freq_nonzero = freq[freq > 0]
    rare_p10 = float(np.percentile(freq_nonzero, 10))
    rare_p25 = float(np.percentile(freq_nonzero, 25))
    
    # Identify rare classes
    rare_classes_p10 = [int(i) for i in range(80) if freq[i] > 0 and freq[i] < rare_p10]
    rare_classes_p25 = [int(i) for i in range(80) if freq[i] > 0 and freq[i] < rare_p25]
    
    print("\n" + "="*60)
    print("COCO2014 Class Frequency Statistics")
    print("="*60)
    print(f"Frequency range: {freq.min()} - {freq.max()}")
    print(f"Mean: {freq.mean():.1f}, Median: {np.median(freq):.1f}")
    print(f"\nPercentile Thresholds:")
    print(f"  P10 (极稀有): {rare_p10:.0f} ({len(rare_classes_p10)} classes)")
    print(f"  P25 (稀有):   {rare_p25:.0f} ({len(rare_classes_p25)} classes)")
    
    print(f"\n极稀有类 (freq < P10):")
    for idx in rare_classes_p10:
        print(f"  [{idx:2d}] {COCO_CLASSES[idx]}: {freq[idx]}")
    
    print(f"\n稀有类 (P10 <= freq < P25):")
    for idx in rare_classes_p25:
        if idx not in rare_classes_p10:
            print(f"  [{idx:2d}] {COCO_CLASSES[idx]}: {freq[idx]}")
    
    print(f"\n高频类 Top-5:")
    top_indices = np.argsort(freq)[::-1][:5]
    for idx in top_indices:
        print(f"  [{idx:2d}] {COCO_CLASSES[idx]}: {freq[idx]}")
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cooccur_coco.json')
    
    result = {
        'dataset': 'COCO2014',
        'num_classes': 80,
        'class_names': COCO_CLASSES,
        'freq': freq.tolist(),
        'cooccur': cooccur.tolist(),
        'rare_p10': rare_p10,
        'rare_p25': rare_p25,
        'rare_classes_p10': rare_classes_p10,
        'rare_classes_p25': rare_classes_p25,
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f)
    
    print(f"\n✓ Saved to: {output_path}")
    print("="*60)

if __name__ == '__main__':
    main()

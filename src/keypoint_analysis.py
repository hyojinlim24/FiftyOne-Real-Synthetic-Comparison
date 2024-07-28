import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_keypoint_data(dataset):
    keypoints_data = defaultdict(list)
    for sample in dataset:
        keypoints = sample.BKP_Landmark_Keypoints
        if keypoints:
            for keypoint in keypoints.keypoints:
                points = keypoint.points
                for idx, point in enumerate(points):
                    if point[0] != 'NAN' and point[1] != 'NAN':
                        x, y = float(point[0]), float(point[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:  # Filter points within (0, 1) range
                            keypoints_data[idx].append((x, y))
    return keypoints_data

def plot_keypoint_scatter(dataset, keypoint_labels, dataset_name):
    for idx, label in enumerate(keypoint_labels):
        points = np.array(dataset[idx])
        if points.size > 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
            plt.title(f'{label} ({dataset_name})')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, 1)  # Set x-axis limit
            plt.ylim(0, 1)  # Set y-axis limit
            plt.show()
        else:
            print(f"No data to plot for keypoint: {label}")

import os
import pandas as pd
from PIL import Image
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_annotations(annotations_path):
    annotations = []
    for file in os.listdir(annotations_path):
        if file.endswith(".txt"):
            image_id = file.split(".txt")[0]
            with open(os.path.join(annotations_path, file), 'r') as f:
                for line in f:
                    values = line.strip().split()
                    class_id = int(values[0])
                    x_center, y_center, width, height = map(float, values[1:])
                    annotations.append([image_id, class_id, x_center, y_center, width, height])
    return pd.DataFrame(annotations, columns=["image_id", "class_id", "x_center", "y_center", "width", "height"])

def findImageSizes(images_path):
    # List to store image sizes
    image_sizes = []

    for file_name in os.listdir(images_path):
        if file_name.endswith(('.jpg')): 
            image_path = os.path.join(images_path, file_name)
            with Image.open(image_path) as img:
                width, height = img.size
                image_sizes.append((file_name, width, height))

    
    df_sizes = pd.DataFrame(image_sizes, columns=["image_id", "width", "height"])

    # Display some statistics
    print(df_sizes.describe())

    print(df_sizes.groupby(['width', 'height']).size().sort_values(ascending=False).head())


    # Group by width and height pairs and count their frequencies
    size_counts = df_sizes.groupby(['width', 'height']).size().reset_index(name='count')

    # Convert width and height to a combined string for easier plotting
    size_counts['size'] = size_counts['width'].astype(str) + 'x' + size_counts['height'].astype(str)

    # Sort by frequency and select the top 6 pairs
    top_6_sizes = size_counts.sort_values(by='count', ascending=False).head(6)

    # Plot the histogram for the top 6 width-height pairs
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_6_sizes, x='size', y='count', color="blue")
    plt.xticks(rotation=45)
    plt.title('Top 6 Most Frequent Width-Height Pairs in Images')
    plt.xlabel('Width x Height (pixels)')
    plt.ylabel('Frequency')
    plt.show()

def heatmap(dir_path):
    labels_path = os.path.join(dir_path, "labels")
    
    heatmap = None

    # Heatmap grid dimensions (e.g., 100x100 for normalized space)
    grid_size = 100
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Process each label file
    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)

        # Read the label file
        with open(label_path, "r") as file:
            print('Processing:', label_file)
            for line in file:
                # YOLO format: class x_center y_center width height
                _, x_center, y_center, _, _ = map(float, line.split())

                # Map normalized coordinates to grid indices
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)

                # Clamp grid indices to ensure they're within bounds
                grid_x = max(0, min(grid_x, grid_size - 1))
                grid_y = max(0, min(grid_y, grid_size - 1))

                # Increment the heatmap at the grid location
                heatmap[grid_y, grid_x] += 1

    # Normalize the heatmap for better visualization
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

        # Display the heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap="hot", interpolation="nearest", extent=[0, 1, 0, 1])
        plt.colorbar(label="Density")
        plt.title("Object Center Density Heatmap (Normalized Coordinates)")
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.savefig('paper/data/heatmap.png')
        plt.show()
    else:
        print("No valid data found.")


if __name__ == "__main__":
    # findImageSizes('uniform/images')
    heatmap('uniform')
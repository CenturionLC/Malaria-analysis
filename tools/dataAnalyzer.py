import os
import pandas as pd
from PIL import Image
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

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
    image_sizes = []

    for file_name in os.listdir(images_path):
        if file_name.endswith(('.jpg')): 
            image_path = os.path.join(images_path, file_name)
            with Image.open(image_path) as img:
                width, height = img.size
                image_sizes.append((file_name, width, height))

    
    df_sizes = pd.DataFrame(image_sizes, columns=["image_id", "width", "height"])

    print(df_sizes.describe())

    print(df_sizes.groupby(['width', 'height']).size().sort_values(ascending=False).head())


    # Group by width and height pairs and count their frequencies
    size_counts = df_sizes.groupby(['width', 'height']).size().reset_index(name='count')

    # Convert width and height to a combined string for easier plotting
    size_counts['size'] = size_counts['width'].astype(str) + 'x' + size_counts['height'].astype(str)

    top_6_sizes = size_counts.sort_values(by='count', ascending=False).head(6)

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

    grid_size = 100
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)

        with open(label_path, "r") as file:
            print('Processing:', label_file)
            for line in file:
                # YOLO format class x_center y_center width height
                _, x_center, y_center, _, _ = map(float, line.split())

                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)

                grid_x = max(0, min(grid_x, grid_size - 1))
                grid_y = max(0, min(grid_y, grid_size - 1))

                heatmap[grid_y, grid_x] += 1

    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

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

def parse_yolo_label(label_path, img_width, img_height):
    boxes = []
    class_ids = []
    box_dims = []
    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])

            # YOLO format to pixel format
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)


            boxes.append((class_id, x_min, y_min, x_max, y_max))
            class_ids.append(class_id)
            box_dims.append((width, height))
    return boxes,class_ids, box_dims


def process_dataset(image_folder, label_folder):
    class_counts = Counter()
    all_boxes = []
    no_label_images = []
    total_labels = 0

    label_files = sorted(os.listdir(label_folder))
    for label_file in label_files:
        img = cv2.imread(os.path.join(image_folder, os.path.splitext(label_file)[0] + ".jpg"))
        img_height, img_width, _ = img.shape

        label_path = os.path.join(label_folder, label_file)
        _,class_ids, box_dims = parse_yolo_label(label_path, img_width, img_height)
        
        if class_ids:
            class_counts.update(class_ids)
            total_labels += len(class_ids)
            all_boxes.extend(box_dims)
        else:
            image_name = os.path.splitext(label_file)[0] + ".jpg"
            no_label_images.append(image_name)

    # Bounding box stats
    box_areas = [w * h for w, h in all_boxes]
    box_widths = [w for w, h in all_boxes]
    box_heights = [h for w, h in all_boxes]

    # Print stats
    print(f"Total number of images: {len(os.listdir(image_folder))}")
    print(f"Total number of label files: {len(label_files)}")
    print(f"Total bounding boxes: {total_labels}")
    print(f"Number of images without labels: {len(no_label_images)}\n")

    print("Class distribution:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} instances")

    print("\nBounding box statistics:")
    print(f"  Average width: {np.mean(box_widths):.4f}")
    print(f"  Average height: {np.mean(box_heights):.4f}")
    print(f"  Average area: {np.mean(box_areas):.4f}")
    print(f"  Median area: {np.median(box_areas):.4f}")
    print(f"  Total bounding box area (normalized): {np.sum(box_areas):.4f}")

    stats_df = pd.DataFrame.from_dict(class_counts, orient="index", columns=["Instances"])
    stats_df.index.name = "Class ID"
    stats_df.to_csv("class_distribution.csv")

def visualize_images(image_folder, label_folder):
    # Load 6 images with their labels
    image_files = sorted(os.listdir(image_folder))[:6]

    plt.figure(figsize=(15, 10))
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        img_height, img_width, _ = img.shape
        if os.path.exists(label_path):
            boxes = parse_yolo_label(label_path, img_width, img_height)
            for class_id, x_min, y_min, x_max, y_max in boxes:
                # Draw bounding boxes and labels
                color = (255, 0, 0)  # Red color for bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Plot image
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img)
        plt.title(f"Image: {image_file}")
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle("Visualizing Images with Bounding Boxes", fontsize=16)
    plt.savefig('paper/data/plot_initial_data.png')
    plt.show()


if __name__ == "__main__":
    # findImageSizes('uniform/images')
    # heatmap('uniform')
    # visualize_images('uniform/images', 'uniform/labels')
    process_dataset('uniform/images', 'uniform/labels')
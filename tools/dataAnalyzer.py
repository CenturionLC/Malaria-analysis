import os
import pandas as pd
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

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


# # Assuming images are of the same dimensions for simplicity (e.g., 640x640)
# image_width, image_height = 640, 640

# # Convert normalized coordinates to pixel coordinates
# df["x_pixel"] = df["x_center"] * image_width
# df["y_pixel"] = df["y_center"] * image_height


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Set up the plot
# plt.figure(figsize=(8, 8))
# sns.kdeplot(
#     x=df["x_pixel"], y=df["y_pixel"],
#     cmap="Reds", fill=True, thresh=0.05, bw_adjust=0.5
# )
# plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
# plt.title("Heatmap of Bounding Box Center Points")
# plt.xlabel("X Position (pixels)")
# plt.ylabel("Y Position (pixels)")
# plt.show()


if __name__ == "__main__":
    findImageSizes('uniform/images')
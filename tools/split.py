import os
import shutil
from sklearn.model_selection import train_test_split

import os

def rename_folders(path, old_name, new_name):
    # List all items in the directory
    for folder in os.listdir(path):
        # Construct the full folder path
        folder_path = os.path.join(path, folder)
        # Check if it’s a folder and matches the old_name
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, old_name)):
            # Define the new folder name and path
            new_folder_name = folder.replace(old_name, new_name)
            new_folder_path = os.path.join(path, new_folder_name)
            # Rename the folder
            os.rename(folder_path, new_folder_path)
            print(f'Renamed "{folder}" to "{new_folder_name}"')



# Just move everything to output folder without any split
def generate_base(input_folder, output_folder):
    # Create output directories
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

    # Function to copy files to the new folders
    def copy_files(source_folder, dest_folder):
        for file_name in os.listdir(source_folder):
            shutil.copy(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))

    # Copy images and labels to their respective folders
    copy_files(os.path.join(input_folder, "images"), os.path.join(output_folder, "images"))
    copy_files(os.path.join(input_folder, "labels"), os.path.join(output_folder, "labels"))

    print("Data copied successfully!")

def split(input_folder, output_folder):
    images_folder = os.path.join(input_folder, "images")
    labels_folder = os.path.join(input_folder, "labels")

    # Create output directories
    for folder in ["images", "labels"]:
        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(output_folder, folder, split), exist_ok=True)

    # Get list of files
    image_files = sorted(os.listdir(images_folder))
    label_files = sorted(os.listdir(labels_folder))

    # Split the data into train, test, and validation sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.25, random_state=42
    )

    # Function to copy files to the new folders
    def copy_files(file_list, source_folder, dest_folder):
        for file_name in file_list:
            shutil.copy(os.path.join(source_folder, file_name), os.path.join(dest_folder, file_name))

    # Copy images and labels to their respective folders
    copy_files(train_images, images_folder, os.path.join(output_folder, "images", "train"))
    copy_files(val_images, images_folder, os.path.join(output_folder, "images", "val"))
    copy_files(test_images, images_folder, os.path.join(output_folder, "images", "test"))
    copy_files(train_labels, labels_folder, os.path.join(output_folder, "labels", "train"))
    copy_files(val_labels, labels_folder, os.path.join(output_folder, "labels", "val"))
    copy_files(test_labels, labels_folder, os.path.join(output_folder, "labels", "test"))

    print("Data split and copied successfully!")





if __name__ == "__main__":
    annotations_folder = "temp/Ghana/Thick"
    output_folder = "dataset"

    split(annotations_folder, output_folder)
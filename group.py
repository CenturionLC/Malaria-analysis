import os
import shutil
import rarfile

def extract_rar_to_temp(rar_file_folder):
    # Identify the first part of the multi-part archive
    rar_files = [f for f in os.listdir(rar_file_folder) if f.endswith('.rar')]
    rar_files.sort()  # Sort to ensure we start with the first part

    first_rar_file = next((f for f in rar_files if 'part1' in f or f.endswith('.rar')), None)
    
    if not first_rar_file:
        raise FileNotFoundError("No first volume of multi-part RAR archive found.")

    rar_file_path = os.path.join(rar_file_folder, first_rar_file)
    
    # Create 'temp' directory if it doesn't exist
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)

    # Open and extract the .rar file starting from the first volume
    with rarfile.RarFile(rar_file_path) as rf:
        rf.extractall(temp_dir)
    print(f"Extracted {rar_file_path} and all parts to {temp_dir}")

def rename_files_with_prefix(folder_path, prefix):
    # Loop through each file in the folder and rename
    for filename in os.listdir(folder_path):
        # Construct old and new file paths
        old_path = os.path.join(folder_path, filename)
        new_name = f"{prefix}{filename}"
        new_path = os.path.join(folder_path, new_name)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_name}")

# Usage
# Use current directory
# rar_file_folder = os.getcwd()
# extract_rar_to_temp(rar_file_folder)


images_folder = 'temp/Ghana/Thick/images'
labels_folder = 'temp/Ghana/Thick/labels'

# Prefix to add
prefix = "thick_ghana_"

# Rename files in both folders
rename_files_with_prefix(images_folder, prefix)
rename_files_with_prefix(labels_folder, prefix)
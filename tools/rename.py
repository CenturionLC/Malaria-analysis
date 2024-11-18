import os

def rename_files_with_prefix(folder_path, prefix):
    # Loop through each file in the folder and rename
    for filename in os.listdir(folder_path):
        # Construct old and new file paths
        if filename.startswith(prefix):
            continue
        old_path = os.path.join(folder_path, filename)
        new_name = f"{prefix}{filename}"
        new_path = os.path.join(folder_path, new_name)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_name}")



def clean_repeated_prefix(folder_path, prefix):
    for filename in os.listdir(folder_path):
        # Check if the filename contains the prefix more than once
        if filename.count(prefix) > 1:
            # Remove all instances of the prefix, then add it back once at the beginning
            new_name = prefix + filename.replace(prefix, "")
            
            # Construct old and new file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            # Rename file only if the name has changed
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed {filename} to {new_name}")


def main(images_folder, labels_folder,prefix):

    # Rename files in both folders
    rename_files_with_prefix(images_folder, prefix)
    rename_files_with_prefix(labels_folder, prefix)

if __name__ == "__main__":
    images_folder = 'uniform/images'
    labels_folder = 'uniform/labels'
    
    prefix = "thick_ghana"

    # main(images_folder, labels_folder, prefix)
    clean_repeated_prefix(images_folder, prefix)
    # clean_repeated_prefix(labels_folder, prefix)
import os

def filter_and_relabel_yolo_labels():
    temp_dir = 'temp'
    labels_dir = os.path.join(temp_dir, 'labels')
    
    # Map for relabeling: 'trophozoite' -> 0, 'white blood cell' -> 1
    label_map = {
        'trophozoite': '0',
        'white blood cell': '1'
    }
    
    # Classes we want to keep
    classes_to_keep = ['trophozoite', 'white blood cell']
    
    # Read the labels.txt file to get the index of each class name
    labels_path = os.path.join(temp_dir, 'label.txt')
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Find indexes of classes we want to keep
    keep_indexes = {str(class_names.index(name)): label_map[name] for name in classes_to_keep}
    
    # Process each label file in the labels directory
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        
        # Read original content
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Filter and relabel lines
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            
            if class_id in keep_indexes:
                # Replace class_id with the new label and keep the rest of the line as is
                new_line = f"{keep_indexes[class_id]} {' '.join(parts[1:])}"
                new_lines.append(new_line)
        
        # Rewrite file with filtered and relabeled data
        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')
        
        print(f"Processed {label_file}")

# Usage
filter_and_relabel_yolo_labels()

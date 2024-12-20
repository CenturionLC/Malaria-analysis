import os

def filter_and_relabel_ghana(dir):
    labels_dir = os.path.join(dir, 'labels')
    
    # Map for relabeling: 'trophozoite' -> 0, 'white blood cell' -> 1
    label_map = {
        'trophozoite': '0',
        'white blood cell': '1'
    }
    
    # Classes we want to keep
    classes_to_keep = ['trophozoite', 'white blood cell']
    
    labels_path = os.path.join(dir, 'label.txt')
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Find indexes of classes we want to keep
    keep_indexes = {str(class_names.index(name)): label_map[name] for name in classes_to_keep}
    
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Filter and relabel lines
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = parts[0]
            
            if class_id in keep_indexes:
                new_line = f"{keep_indexes[class_id]} {' '.join(parts[1:])}"
                new_lines.append(new_line)

        # Remove duplicates 
        new_lines = list(set(new_lines))
        
        with open(label_path, 'w') as f:
            if len(new_lines) == 0:
                f.write('')
            else: 
                f.write('\n'.join(new_lines) + '\n')
        
        print(f"Processed {label_file}")

def filter_and_relabel_uganda(dir):
    # Uganda has Trop as index 0 and WBC as index 1 already
    labels_dir = os.path.join(dir, 'labels')

    keep_indexes = ['0','1']

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_name = parts[0]

            if class_name in keep_indexes:
                new_line = f"{class_name} {' '.join(parts[1:])}"
                new_lines.append(new_line)

        # Remove duplicates 
        new_lines = list(set(new_lines))
        
        with open(label_path, 'w') as f:
            if len(new_lines) == 0:
                f.write('')
            else: 
                f.write('\n'.join(new_lines) + '\n')

        print(f"Processed {label_file}")



# Usage
if __name__ == '__main__':
    filter_and_relabel_uganda('temporary/uganda')

# pylint: disable=all

# %%==================== CHECK GPU AVAILABILITY ====================%%
import torch
from pathlib import Path

# print(torch.cuda.is_available())  # Should return True if GPU is available
# print(torch.cuda.current_device())  # Should return the current GPU device

# num_gpus = torch.cuda.device_count()
# print(f"Number of GPUs available: {num_gpus}")

# # Loop through available GPUs and print their names and details
# for i in range(num_gpus):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
#     print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
#     print(f"  Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
import gdown



DATA_DIR = Path('Lacuna')



def download_from_drive(id, name: str):
    output_path = DATA_DIR / name

    if output_path.exists():
        print(f"{name} already exists in {output_path}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}...")
    gdown.download(
        f"https://drive.google.com/uc?id={id}", str(output_path), quiet=False
    )

    return output_path
download_from_drive("1MBNfQop9wMcZrl5XLiQDuQGYm-Yl86c-", "images.zip")

# %%==================== IMPORT LIBRARIES ====================%%
import pandas as pd
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
import multiprocessing

# %%==================== DEFINE PATHS ====================%%
# Path to where your data is stored
DATA_DIR = Path('Lacuna')

# Preview data files available
os.listdir(DATA_DIR)
# Set up directoris for training a yolo model

# Images directories
DATASET_DIR = Path('datasets/dataset')
IMAGES_DIR = DATASET_DIR / 'images'
TRAIN_IMAGES_DIR = IMAGES_DIR / 'train'
VAL_IMAGES_DIR = IMAGES_DIR / 'val'
TEST_IMAGES_DIR = IMAGES_DIR / 'test'

# Labels directories
LABELS_DIR = DATASET_DIR / 'labels'
TRAIN_LABELS_DIR = LABELS_DIR / 'train'
VAL_LABELS_DIR = LABELS_DIR / 'val'
TEST_LABELS_DIR = LABELS_DIR / 'test'

# Set to false to (re)initialize data structure
SKIP_RECREATE_DIRS = True

# %%==================== UNZIP IMAGES ====================%%
if not os.path.exists('images'):
    shutil.unpack_archive(DATA_DIR / 'images.zip', 'images')

# %%==================== LOAD DATASETS ====================%%
# Train & test data
train = pd.read_csv(DATA_DIR / 'Train.csv')
test = pd.read_csv(DATA_DIR / 'Test.csv')
ss = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

# Add image_path column
train['image_path'] = [Path('images/' + x) for x in train.Image_ID]
test['image_path'] = [Path('images/' + x) for x in test.Image_ID]

# Map str classes to ints (label encoding targets)
train['class_id'] = train['class'].map({'Trophozoite': 0, 'WBC': 1, 'NEG': 2})

# Preview the head of the train set
train.head()


# %%==================== SPLIT INTO TRAINING & VALIDATION ====================%%
train_unique_imgs_df = train.drop_duplicates(subset = ['Image_ID'], ignore_index = True)
X_train, X_val = train_test_split(train_unique_imgs_df, test_size = 0.25, stratify=train_unique_imgs_df['class'], random_state=42)

X_train = train[train.Image_ID.isin(X_train.Image_ID)]
X_val = train[train.Image_ID.isin(X_val.Image_ID)]

# Check shapes of training and validation data
print(f"X Train shape: {X_train.shape} ; X Val shape: {X_val.shape}")

# %%==================== PREVIEW TARGET DISTRIBUTION ====================%%
# TODO: Handle class imbalance
X_train['class'].value_counts(normalize = True), X_val['class'].value_counts(normalize = True)

# %%==================== INTIIALIZE IMAGE & LABEL DIRECTORIES ====================%%
IMAGE_DIRS = [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR, TRAIN_LABELS_DIR,VAL_LABELS_DIR, TEST_LABELS_DIR]

allDirsExist = all([d.exists() for d in IMAGE_DIRS])

if (allDirsExist and SKIP_RECREATE_DIRS):
    print("Skipping recreation of image & label directories")
else:
    #----- (Re)create all dirs as empty -----#
    for DIR in IMAGE_DIRS:
        if DIR.exists():
            shutil.rmtree(DIR)
        DIR.mkdir(parents=True, exist_ok = True)

    #----- Copy images to their respective directories -----#
    print("Copying images")
    for img in tqdm(X_train.image_path.unique()):
        shutil.copy(img, TRAIN_IMAGES_DIR / img.parts[-1])

    for img in tqdm(X_val.image_path.unique()):
        shutil.copy(img, VAL_IMAGES_DIR / img.parts[-1])

    for img in tqdm(test.image_path.unique()):
        shutil.copy(img, TEST_IMAGES_DIR / img.parts[-1])

    #----- Write labels to their respective directories -----#
    print("Writing labels")
    # Convert bboxes to yolo format & save them
    def save_yolo_annotation(row):
        image_path, class_id, output_dir = row['image_path'], row['class_id'], row['output_dir']

        label_file = Path(output_dir) / f"{Path(image_path).stem}.txt"

        if class_id == 2:
            # If class is NEG, create an empty label file
            open(label_file,'w').close()
            return

        img = cv2.imread(image_path)

        # print("IMAGE:", img)

        if img is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        height, width, _ = img.shape

        ymin, xmin, ymax, xmax = row['ymin'], row['xmin'], row['ymax'], row['xmax']

        # Normalize the coordinates
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        with open(label_file, 'a') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    # Save train and validation labels
    for df, outDir in [(X_train, TRAIN_LABELS_DIR), (X_val, VAL_LABELS_DIR)]:
        df['output_dir'] = outDir
        with multiprocessing.Pool() as pool:
            list(tqdm(pool.imap(save_yolo_annotation, df.to_dict('records')), total=(len(X_train) + len(X_val))))

# %%==================== CREATE DATA.YAML FROM CONF.JSON ====================%%
from json import loads

# Load conf
config = { "train": "./images/train", "val": "./images/val", "test": "./images/test" } # Fallback
if os.path.exists('conf.json'):
    with open('conf.json', 'r') as f:
        config = config | loads(f.read())

print(os.getcwd())

# Create a data.yaml file required by yolo
class_names = ['Trophozoite', 'WBC','NEG']
num_classes = len(class_names)

data_yaml = {
    'train': config['train'],
    'val': config['val'],
    'test': config['test'],
    'nc': num_classes,
    'names': class_names
}

# Write to file
yaml_path = 'data.yaml'
with open(yaml_path, 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)

# %%==================== PLOT TRAINING IMAGES WITH BBOXES ====================%%
# Plot some images and their bboxes to ensure the conversion was done correctly
def load_annotations(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

# Plot an image annotated with its bboxes
def plot_image_with_boxes(image_path, boxes):
    # Load the image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    h, w, _ = image.shape

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot each bounding box
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # Convert YOLO format to corner coordinates
        xmin = int((x_center - width / 2) * w)
        ymin = int((y_center - height / 2) * h)
        xmax = int((x_center + width / 2) * w)
        ymax = int((y_center + height / 2) * h)

        # Draw the bounding box
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                          edgecolor='red', facecolor='none', linewidth=2))
        plt.text(xmin, ymin - 10, f'Class {int(class_id)}', color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

# Directories for images and labels
IMAGE_DIR = TRAIN_IMAGES_DIR
LABEL_DIR = TRAIN_LABELS_DIR

# Plot a few images with their annotations
for image_name in os.listdir(IMAGE_DIR)[:3]:
    image_path = IMAGE_DIR / image_name
    label_path = LABEL_DIR / (image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    if label_path.exists():
        boxes = load_annotations(label_path)
        print(f"Plotting {image_name} with {len(boxes)} bounding boxes.")
        plot_image_with_boxes(image_path, boxes)
    else:
        print(f"No annotations found for {image_name}.")


# %%==================== LOAD & FINE-TUNE YOLO ====================%%
# Load a yolo pretrained model
# model = YOLO('yolo11l.pt')
# detect  = Path(os.getcwd()) / "runs/detect"



# # Fine tune model to our data
results = model.train(
    data='data.yaml',          # Path to the dataset configuration
    epochs=40,                 # Number of epochs
    imgsz= 1280,       # Image size (height, width) randomly pick betwen 640 and 1280
    batch=8,                   # Batch size
    device=0,                  # Device to use (0 for the first GPU)
    patience=10,
    dropout=0.1,
    lrf=0.1,
    project=detect,
    name="run",
)


# %%  


# # Takes the latest run's best weights
model = YOLO("runs/detect/run9/weights/best.pt")   # Load trained YOLO model


test_dir_path = 'datasets/dataset/images/test'      # Path to the test images directory
image_files = os.listdir(test_dir_path)
all_data = []                                       # Results for all images

# For all test images
for image_file in tqdm(image_files):
    img_path = os.path.join(test_dir_path, image_file)

    img_results = model(
        img_path,
    )

    # Extract results
    boxes = img_results[0].boxes.xyxy.tolist()          # Bounding boxes in xyxy format
    classes = img_results[0].boxes.cls.tolist()         # Class indices
    confidences = img_results[0].boxes.conf.tolist()    # Confidence scores
    names = img_results[0].names                        # Class names dictionary

    if not boxes:
        # No detections => class is NEG
        all_data.append({
            'Image_ID': image_file,
            'class': 'NEG',
            'confidence': 1.0,
            'ymin': 0, 'xmin': 0, 'ymax': 0, 'xmax': 0
        })
    else:
        # For all detections
        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confidences):
            all_data.append({
                'Image_ID': image_file,
                'class': names[int(cls)],
                'confidence': conf,
                'ymin': y1, 'xmin': x1, 'ymax': y2, 'xmax': x2
            })

# # # Convert the list to a DataFrame for all images
sub = pd.DataFrame(all_data)
sub.to_csv('benchmark_submission.csv', index = False)

# %%

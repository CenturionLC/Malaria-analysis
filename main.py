
import torch
from ultralytics import YOLO
from tqdm.notebook import tqdm
import pandas as pd
import os
from pathlib import Path
import yaml

from tools.setup import setupProject


def run():
    if not os.path.exists('dataset'):
        setupProject()

    DATASET_DIR = Path('dataset')
    IMAGES_DIR = DATASET_DIR / 'images'

    TRAIN_IMAGES_DIR = IMAGES_DIR / 'train'
    VAL_IMAGES_DIR = IMAGES_DIR / 'val'
    TEST_IMAGES_DIR = IMAGES_DIR / 'test'

    from json import loads
    config = { "train": "./images/train", "val": "./images/val", "test": "./images/test" } # Fallback
    if os.path.exists('conf.json'):
        with open('conf.json', 'r') as f:
            config = config | loads(f.read())

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

    yaml_path = 'data.yaml'
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

    device = torch.device(
        0
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO("transferred.pt")

    detect  = Path(os.getcwd()) / "runs/detect"

    results = model.train(
        data='data.yaml',
        epochs=30,
        batch=4,
        imgsz="1024,768",
        device=device,
        cache="disk",
        project=detect,
        name="run",
    )


# # Grab save dir from dictionary
# best_weights_path = results.save_dir / 'weights/best.pt'
# last_weights_path = results.save_dir / 'weights/last.pt'   


# # Load model for validation
# model = YOLO("runs/detect/run14/weights/last.pt")


# print('\nPerforming testing...\n')


# results = model.val(
#     data='data.yaml',
#     project="runs/detect/run9/",
#     name="test",
#     iou=0.3,
#     imgsz=1280,
#     split='test',
#     conf=0.001,
# )



if __name__ == "__main__":
    run()



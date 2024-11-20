
import torch
from ultralytics import YOLO
from tqdm.notebook import tqdm
import pandas as pd
import os
from pathlib import Path
import yaml

from tools.setup import setupProject


def run(device):
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

    # Create the data.yaml file
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



    model = YOLO("transferred.pt")

    detect  = Path(os.getcwd()) / "runs/detect"

    model.train(
        data='data.yaml',
        epochs=30,
        batch=4,
        imgsz=1024,
        device=device,
        cache="disk",
        project=detect,
        name="run",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
    )

def test(device):
    model = YOLO('best.pt')

    print('\nPerforming testing...\n')


    model.val(
        data='data.yaml',
        project="runs/detect/",
        name="test",
        device=device,
        iou=0.3,
        imgsz=1024,
        split='test',
        conf=0.01,
        save_json=True,
    )


if __name__ == "__main__":
    device = torch.device(
        0
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    # run(device)
    test(device)



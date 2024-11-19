# Optimization of YOLO Architecture for Parasite Detection and Classification of Malaria in Microscopic Blood Smears


![Alt text](./plot_initial_data.png "Image Title")

## Aims 

This repository aims to analyze and perform malaria classification on the Makerere AI Lab Lacuna Malaria Dataverse available here : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VEADSE

This repository will explore the dataset, perform some analysis and build a model that can classify malaria in blood smears. The YOLO (You only look once) model will be used to assist with the 
detection and classification of the malaria in the blood smears.

## Specifications

This project was built in python 3.8.5 and uses a wide array of libraries such as : 

- numpy
- pandas
- matplotlib
- ultralytics

A full list of the dependencies can be found in the requirements.txt file.

## Technical Requirements

The model was trained on a L40S GPU with 48 GB of dedicated VRAM on an image size of 1024, batch size of 8 using the medium sized YOLOv11 (m) model. 

Running the model will require a very large amount of RAM ~ 16GB dedicated VRAM and should take about 2-3 hours to train for 30 epochs. 

The codebase was run and hosted on  https://lightning.ai and should only cost about $20 to train the model for 30 epochs.

## Getting started

To get started with this project, clone the repository and download the dataset from the link above.

Please move each .rar from the dataset into a folder named datasets in the root directory of the project. 

To install the dependencies for this project, you can run the following if you are using pip : 

- pip install -r requirements.txt

You will need to add a conf.json to the root of the project that defines the path to the yolo dataset for yolo to register it. The json looks as follows : 


```json
{
    "train": "{path_to_repo}/dataset/images/train",
    "val": "{path_to_repo}/dataset/images/val",
    "test": "{path_to_repo}/dataset/images/test"
}
```

Kindly note that the dataset, images and the train,val and test folders will all be generated once the main.py is run. 

You can then simply run the following command to extract the dataset, amend it and then train and test the model :

```bash
    python3 main.py
```

## Results

YOLO will generate a runs folder within the root of the project that contains the results of the training and testing of the model.

More fine-grained and precise data analysis output can be found by making use of the scripts in the tools folder. 
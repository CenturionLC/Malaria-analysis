

import os

from tools.extractor import main as extractFiles
from tools.split import split, rename_folders, generate_base
from tools.relabel import filter_and_relabel_ghana,filter_and_relabel_uganda

def setupProject():

    if not os.path.exists('datasets'):
        # throw 
        print('Please download the datasets and place in a folder named datasets.')
        exit(1)

    # Extract files if not yet extracted
    if not os.path.exists('temporary'):
        extractFiles()
        print('Files extracted successfully.')

    ### Rename folders 

    rename_folders('temporary/ghana/Ghana/Thick', 'labels_yolo', 'labels')
    rename_folders('temporary/ghana/Thin Images Ghana/Thin Images With Annotations', 'labels_yolo', 'labels')
    rename_folders('temporary/uganda', 'Labels-YOLO', 'labels')

    print('Folders renamed successfully.')

    ### Relabel data
    filter_and_relabel_ghana('temporary/ghana/Thin Images Ghana/Thin Images With Annotations')
    filter_and_relabel_uganda('temporary/uganda')
    
    # Rename files to be unique in the dataset
    from tools.rename import main as rename_files_with_prefix

    # Thick Ghana
    prefix = "thick_ghana_"
    input_folder = 'temporary/ghana/Ghana/Thick/images'
    label_folder = 'temporary/ghana/Ghana/Thick/labels'

    rename_files_with_prefix(input_folder,label_folder, prefix)

    # Thin Ghana
    prefix = "thin_ghana_"
    input_folder = 'temporary/ghana/Thin Images Ghana/Thin Images With Annotations/images'
    label_folder = 'temporary/ghana/Thin Images Ghana/Thin Images With Annotations/labels'

    rename_files_with_prefix(input_folder,label_folder, prefix)

    # uganda

    prefix = "uganda_"

    input_folder = 'temporary/uganda/images'
    label_folder = 'temporary/uganda/labels'

    rename_files_with_prefix(input_folder,label_folder, prefix)

    print('Files renamed successfully.')

    ### Move files to dataset folder
    print('Splitting data into train, test, and validation sets...')

    input_folder = 'temporary/ghana/Ghana/Thick'
    output_folder = 'dataset'

    split(input_folder, output_folder)

    input_folder = 'temporary/ghana/Thin Images Ghana/Thin Images With Annotations'
    output_folder = 'dataset'

    split(input_folder, output_folder)

    input_folder = 'temporary/uganda'
    output_folder = 'dataset'

    split(input_folder, output_folder)


    print('Data split successfully.')

    ### Create seperate uniform dataset 


    if not os.path.exists('uniform'):
        print('Creating a uniform dataset...')

        generate_base('temporary/ghana/Ghana/Thick', 'uniform')
        generate_base('temporary/ghana/Thin Images Ghana/Thin Images With Annotations', 'uniform')
        generate_base('temporary/uganda', 'uniform')

        print('Uniform dataset created successfully.')


if __name__ == "__main__":
    setupProject()
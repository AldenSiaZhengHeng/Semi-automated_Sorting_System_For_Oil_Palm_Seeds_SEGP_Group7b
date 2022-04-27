'''
This file is provided by our supervisor, Dr Iman Yi Liao.
This file defines functions that create csv files based on annotated dataset.
It includes
    - a csv file for storing image of seed set and its label, i.e., 'GOOD', 'BAD', or 'MIX'
    - a csv file for storing bounding boxes of individual seeds, and their labels, i.e., 'GOOD' or 'BAD'
        and the image file name where the seeds were detected from
'''

# import modules
import csv
import os
import glob
from utils.const import *
from utils.seed_io import read_from_commandline


# Define funciton to create empty csv file
def create_empty_csv(root, filename, header):
    # Create the above CSV files if they do not exist
    filename = os.path.join(root, SUBFOLDER_NAMES['CSV'], filename)
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    return filename


# Define function to create the empty csv file for storing image directory to read from
def create_empty_originaldata_csv(root=CODE_ROOT, filename=CSV_SEEDS_IMAGES):
    header = CSV_SEEDS_HEADER
    return create_empty_csv(root, filename, header)


# Define function to create empty csv files for saving detection results  as annotations
def create_empty_annotation_csv(root=CODE_ROOT, filename=CSV_SEED_INDIVIDUAL_ANNOTATED):
    header = CSV_SEED_INDIVIDUAL_HEADER
    return create_empty_csv(root, filename, header)


# Define function to create empty csv files for saving detection results as predictions
def create_empty_detection_csv(root=CODE_ROOT, filename=CSV_SEED_INDIVIDUAL_PREDICTED):
    header = CSV_SEED_INDIVIDUAL_HEADER
    return create_empty_csv(root, filename, header)


# Define function to create empty csv file for recording testing results
def create_empty_test_record(root=CODE_ROOT, filename=CSV_TEST_RECORD):
    header = CSV_TEST_RECORD_HEADER
    return create_empty_csv(root, filename, header)


# Define function to create empty csv for cropped seed images
def create_empty_cropped_seed_csv(root=CODE_ROOT, filename=CSV_CROPPED_SEEDS):
    header = CSV_CROPPED_SEEDS_HEADER
    return create_empty_csv(root, filename, header)


def check_img_label(imgfile):
    if 'MIX' in imgfile.upper():
        img_label = IMAGE_LABELS['MIX']
        annotation_file = imgfile.split('.')[0] + '.xml'
    elif 'GOOD' in imgfile.upper():
        img_label = IMAGE_LABELS['GOOD']
        annotation_file = imgfile.split('.')[0] + '.xml'
#        annotation_file = None
    elif 'BAD' in imgfile.upper():
        img_label = IMAGE_LABELS['BAD']
        annotation_file = imgfile.split('.')[0] + '.xml'
#        annotation_file = None
    else:
        img_label = IMAGE_LABELS['UNKNOWN']
        annotation_file = None

    return img_label, annotation_file


# Define function to append to cropped seeds csv file
def generate_cropped_seeds_record_from(dataset_path=DATA_ROOT, code_path=CODE_ROOT, outcsv=CSV_CROPPED_SEEDS):
    # Record seeds images in csv_seeds from the folder and subfolders of the given dataset_path
    filename = create_empty_cropped_seed_csv(code_path, outcsv)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # loop through the folder and the subfolders to add into the record
        path = os.path.join(dataset_path, '**/*.png')
        for imgfile in glob.iglob(path, recursive=True):
            img_label, _ = check_img_label(imgfile)
            writer.writerow([imgfile, img_label])
            
        path = os.path.join(dataset_path, '**/*.jpg')
        for imgfile in glob.iglob(path, recursive=True):
            img_label, _ = check_img_label(imgfile)
            writer.writerow([imgfile, img_label])
            
        path = os.path.join(dataset_path, '**/*.JPG')
        for imgfile in glob.iglob(path, recursive=True):
            img_label, _ = check_img_label(imgfile)
            writer.writerow([imgfile, img_label])


# Define function to append to csv_seeds file
def generate_seeds_record_from(dataset_path=DATA_ROOT, code_path=CODE_ROOT, outcsv=CSV_SEEDS_IMAGES):
    # Record seeds images in csv_seeds from the folder and subfolders of the given dataset_path
    filename = create_empty_originaldata_csv(code_path, outcsv)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # loop through the folder and the subfolders to add into the record
        i = 0
        path = os.path.join(dataset_path, '**/*.JPG')
        for imgfile in glob.iglob(path, recursive=True):
            i += 1
            # if Mix then record the annotation file as well
            img_label, annotation_file = check_img_label(imgfile)
            writer.writerow([imgfile, img_label, annotation_file])
            
        path = os.path.join(dataset_path, '**/*.jpg')
        for imgfile in glob.iglob(path, recursive=True):
            i += 1
            # if Mix then record the annotation file as well
            img_label, annotation_file = check_img_label(imgfile)
            writer.writerow([imgfile, img_label, annotation_file])
            
        path = os.path.join(dataset_path, '**/*.png')
        for imgfile in glob.iglob(path, recursive=True):
            i += 1
            # if Mix then record the annotation file as well
            img_label, annotation_file = check_img_label(imgfile)
            writer.writerow([imgfile, img_label, annotation_file])
        

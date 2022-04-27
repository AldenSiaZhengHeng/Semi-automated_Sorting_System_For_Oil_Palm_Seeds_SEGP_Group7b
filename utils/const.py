'''
This file is provided by our supervisor, Dr Iman Yi Liao.
where include all the filepath to read the input, file location and trained model.
# There are a little bit modification had done in this file to match to our system
'''

import torch.nn as nn

# Set the root directory where all the python files for the project and the folders mentioned about should be located
#CODE_ROOT = '/content/drive/My Drive/AAR'
CODE_ROOT = '/home/pi/Desktop/PI_SEGP'
# Set the root directory where all the original dataset are stored
#DATA_ROOT = '/content/drive/My Drive/AAR/dataset'
#DATA_ROOT = '/Users/User/Desktop/AAR/dataset'
DATA_ROOT = '/home/pi/Desktop/PI_SEGP/resources/temp/current_input'

# The subfolders that store dataset, output, models, and runs
SUBFOLDER_NAMES = {'CSV': 'csv', 'OUT': 'out', 'MODELS': 'models', 'RUNS': 'runs', 'FIGURES': 'figures'}

'''
Define function to create csv files that are needed for storing individual seeds
The column header in the csv file for individual oil palm seeds should be as follows
 COLS = ["file_name", "x_min", "y_min", "x_max", "y_max", "bbox_label"],
where bbox_label can be 'GOOD', 'BAD' and bbox_annotated can be 'YES' or 'NO'
'''
# The following two files share the same header structure
CSV_SEED_INDIVIDUAL_ANNOTATED = 'seed_individual_annotated.csv'
CSV_SEED_INDIVIDUAL_PREDICTED = 'seed_individual_predicted.csv'

'''Also create csv file that stores all the image file names that need to go through seed detection/segmentation
The column header for sets of seeds (for testing, i.e., annotated) should be as follows
COLS = ['file_name', 'label'], where label can be 'GOOD', 'BAD', 'MIX', 'UNKNOWN'.
For image labelled as 'GOOD', all seeds detected should be labelled as 'GOOD'
For image labelled as 'BAD', all seeds detected should be labelled as 'BAD'
For image labelled as 'MIX', seeds detected  should be labelled according to the annotations available

For image labelled as 'UNKNOWN', seeds detected should be stored in SEED_CSV_PREDICTED file
'''
CSV_SEEDS_IMAGES = 'seeds.csv'
CSV_TEST_RECORD = 'test_record.csv'
CSV_CROPPED_SEEDS = 'cropped_seeds.csv'

# Define the labels used for seeds images
IMAGE_LABELS = {'GOOD': 'GOOD', 'BAD': 'BAD', 'MIX': 'MIX', 'UNKNOWN': 'UNKNOWN'}
SEED_LABELS = {'GOOD': 'GOOD', 'BAD': 'BAD'}

# Define the column header in the csv file for individual oil palm seeds
# bbox_label stores value 'GOOD' if it's annotated as good seed, otherwise 'BAD'.
CSV_SEED_INDIVIDUAL_HEADER = ["file_name", "x_min", "y_min", "x_max", "y_max", "bbox_label"]
CSV_SEEDS_HEADER = ["img_file_name", "img_file_label", "annotation_file_name"]
CSV_TEST_RECORD_HEADER = ["annotated_test_csv_file_name", "segmentation method", "classification method", "accuracy",
                          "precision", "recall", "F1", "AUC"]
CSV_CROPPED_SEEDS_HEADER = ['file_name', 'label']

# Define constants for parser
PARSER_INPUT = '--inputroot'
PARSER_OUTPUT = '--outputroot'

# Define the default mode to seed detection, i.e., for annotation
#FOR_ANNOTATION = True
FOR_ANNOTATION = False

# Define the default image format provided
IMAGE_FORMAT = 'JPG'

# Define the scale for resizing images
IMAGE_RESIZE_SCALE_PERCENTAGE = 20

# Set the default network model for seed classification, trained model weights, and loss criterion used
NETWORK_MODEL = [
    'G2',
    '/home/pi/Desktop/PI_SEGP/models_old/G2_05-22 11_12_50_out.pth',
    #'/home/pi/Desktop/PI_SEGP/models_old/G2__07-07 14_26_42_out.pth/',
    'edge'
    ]
USE_COLOUR_DESCRIPTOR = False
MODEL_INPUT_SIZE = 256
SEGMENT_TYPE = 'G10'
MASK_TYPE = 'edge'  # it can be one of ['combined', 'edge']

# Define the default parameters for seed classification testing
SINGLE_MODEL = True

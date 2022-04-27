'''
This file is provided by our supervisor, Dr Iman Yi Liao.
This file implements functions to read in parameters from command line
'''

import argparse
from utils.const import *
import cv2
import os


# Define function to read in the data root directory and code root directory from command line
def read_from_commandline():
    # Construct a parser
    parser = argparse.ArgumentParser()
    # Specify the data root directory or take the default from const
    parser.add_argument('--imageinputroot', nargs='?', const=DATA_ROOT, default=DATA_ROOT)
    parser.add_argument('--csvinputroot', nargs='?', const=CODE_ROOT, default=CODE_ROOT)
    # Specify the output root directory or take the default from const
    parser.add_argument('--outputroot', nargs='?', const=CODE_ROOT, default=CODE_ROOT)
    # Specify the input file name if any
    parser.add_argument('-fi', '--fileinput', nargs='?', const=CSV_SEEDS_IMAGES, default=CSV_SEEDS_IMAGES)
    # Specify the output file name if any
    parser.add_argument('-oo', '--image2csvoutput', nargs='?', const=CSV_SEEDS_IMAGES, default=CSV_SEEDS_IMAGES)
    parser.add_argument('-ao', '--annotation2csvoutput', nargs='?', const=CSV_SEED_INDIVIDUAL_ANNOTATED,
                        default=CSV_SEED_INDIVIDUAL_ANNOTATED)
    parser.add_argument('-do', '--detection2csvoutput', nargs='?', const=CSV_SEED_INDIVIDUAL_PREDICTED,
                        default=CSV_SEED_INDIVIDUAL_PREDICTED)
    parser.add_argument('-an', '--annotate', dest='annotation', action='store_true')
    parser.add_argument('-na', '--no-annotate', dest='annotation', action='store_false')
    parser.set_defaults(annotation=FOR_ANNOTATION)
    parser.add_argument('-lb', '--imglabel', nargs='?', type=str, const=None,
                        default=None)  # not to specify which type of images for detection
    parser.add_argument('-mt', '--model_type', nargs='?', default=NETWORK_MODEL[0], const=NETWORK_MODEL[0])
    parser.add_argument('-mf', '--model_file', nargs='?', default=NETWORK_MODEL[1], const=NETWORK_MODEL[1])
    parser.add_argument('-tp', '--test_performance_record', nargs='?', default=CSV_TEST_RECORD, const=CSV_TEST_RECORD)
    parser.add_argument('-st', '--segment_type', nargs='?', default=SEGMENT_TYPE, const=SEGMENT_TYPE)
    parser.add_argument('--mask_type', nargs='?', default=MASK_TYPE, const=MASK_TYPE)
    parser.add_argument('-single', '--single_model', dest='single', action='store_true')
    parser.add_argument('-batch', '--multiple_models', dest='single', action='store_false')
    parser.set_defaults(single=SINGLE_MODEL)
    args = parser.parse_args()

    return args


# define function to display resized image
def display_resized_image(img_input, scale_percent, img_title='Window 1', save2file=False):
    # check if the input is image file or img matrix
    if isinstance(img_input, str):
        img = cv2.imread(img_input, cv2.IMREAD_UNCHANGED)
    else:
        img = img_input
    #    print('Original Dimensions : ', img.shape)
    #    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if save2file:
        currentdir = os.getcwd()
        figure_folder = os.path.join(currentdir, 'figures')
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        cv2.imwrite(os.path.join(figure_folder, img_title), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    #    print('Resized Dimensions : ', resized.shape)
    cv2.imshow(img_title, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

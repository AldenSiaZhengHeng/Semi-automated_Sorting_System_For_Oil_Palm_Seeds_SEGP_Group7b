'''
#This file is provided by our supervisor, Dr Iman Yi Liao.
# There are a little bit modification had done in this file to match to our system
# This file implements functions to detect individual seeds from
# an input image of a set of seeds, and output the image overlaid
# with bounding boxes and a csv file that contains the coordinates
# of the bounding boxes.
#
# By Dr. Iman Yi Liao, 22 June 2021
'''

import numpy as np
import pandas as pd
import csv
import cv2
import os
# Progress bar
from tqdm.notebook import tqdm
from utils.const import *
from utils.seed_io import read_from_commandline, display_resized_image
from create_csv_files import create_empty_annotation_csv, create_empty_detection_csv, check_img_label
from seedxml2csv import xml2csv
from seed_classification import predict_seed, load_trained_model
from preprocess import *
from datetime import datetime


# Define function to determine the colour for drawing bbox
def get_color(label):
    if label == 1 or label == 'GOOD':
        color = (0, 255, 0)  # Green for good
    elif label == 0 or label == 'BAD':
        color = (255, 0, 0)  # Red for bad
    else:
        color = (0, 0, 255)  # Blue for not specified

    return color


# Define function to merge or remove bounding boxes if they are largely overlapping or too small
def filter_bbox(bboxes):
    # bboxes are stored as [x_min, y_min, dx, dy]
    out_bboxes = []
    for i in range(len(bboxes)):
        if bboxes[i][2] > 200 and bboxes[i][3] > 200:
        #if bboxes[i][2] > 100 and bboxes[i][3] > 100:
            out_bboxes.append(bboxes[i])

    return out_bboxes


def segment(imgpath, seed_mask, outcsv, display_segmented=False, display_bound=False, img_label=IMAGE_LABELS['UNKNOWN'],
            for_annotation=FOR_ANNOTATION, model=None):
    '''
    Function for image segmentation
    :param path: str, Image path.
    :param seed_mask: ndarray, Seed mask generated from pre_process function.
    :param display_segmented: bool, default=False, Display segmented seeds.
    :param display_bound: bool, default=False, Display images with seed bounding boxes.
    :param outpath: str, output directory
    :param img_label: str, 'GOOD', 'BAD', 'MIX', or 'UNKNOWN'
    :param for_annotation: bool, true for annotation, false otherwise
    :return:
    '''
    # Find seed edges
    seed_contours, _ = cv2.findContours(seed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # seed_contours_filter = list(filter(lambda contour: (cv2.boundingRect(contour)[2] < 1500) and (cv2.boundingRect(contour)[3] < 1500), seed_contours))
    # seed_contours_10 = sorted(seed_contours_filter, key=cv2.contourArea)[-10:]
    # sorted_seed_contours = sorted(seed_contours_10, key=lambda contour: cv2.boundingRect(contour)[0])
    sorted_seed_contours = sorted(seed_contours, key=lambda contour: cv2.boundingRect(contour)[0])

    # Get seed bounding boxes
    seed_bound = [None] * len(sorted_seed_contours)
    for i, contours in enumerate(sorted_seed_contours):
        seed_bound[i] = cv2.boundingRect(contours)
    seed_bound = filter_bbox(seed_bound)

    # Overlay seed bounding boxes
    image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    image_seed_bound = image.copy()
    bound_padding = 5


    # create folders for storing individual seed images and bbox images if they do not already exsit
    outpath = outcsv.rsplit('/')[0]
    folder_name_cropped = os.path.join(outpath, SUBFOLDER_NAMES['OUT'], 'cropped')
    if not os.path.exists(folder_name_cropped):
        os.makedirs(folder_name_cropped)
    folder_name_bound = os.path.join(outpath, SUBFOLDER_NAMES['OUT'], 'bbox')
    if not os.path.exists(folder_name_bound):
        os.makedirs(folder_name_bound)

    imagename = imgpath.rsplit('/', 2)[2][:-4]
    xmlname = imgpath.split('.')[0] + '.xml'
    # print(xmlname)
    good_seeds = 0
    bad_seeds = 0

    matrix = np.array([0,0,0,0])
    laser_matrix = [[0,0],[0,0]]

    # Draw Bounding box on the image to determine the bad and good seeds
    if for_annotation and os.path.exists(xmlname):
        # call seedxmal2csv
        # print('Calling xml2csv function to generate annotations...')
        rows = xml2csv(xmlname)
        rows = [[i[CSV_SEED_INDIVIDUAL_HEADER[j]] for j in range(len(CSV_SEED_INDIVIDUAL_HEADER))] for i in rows]
        for i in range(len(rows)):
            label = rows[i][5]
            color = get_color(label)
            cv2.rectangle(image_seed_bound,
                          (int(rows[i][1] - bound_padding),
                           int(rows[i][2] - bound_padding)),
                          (int(rows[i][3] + bound_padding),
                           int(rows[i][4] + bound_padding)), color, 2)
            seed_segmented = image[rows[i][2]:rows[i][4], rows[i][1]:rows[i][3]]
            cv2.imwrite(os.path.join(folder_name_cropped, imagename + '_' + str(i) + '.jpg'),
                        cv2.cvtColor(seed_segmented, cv2.COLOR_RGB2BGR))
    else:  # not for annotation, or for annotation but annotation file (.xml) does not exist
        rows = []
        #for i in range(len(seed_bound)):

        for i in range(4):
            # cropped it out
            seed_segmented = image[seed_bound[i][1]:seed_bound[i][1] + seed_bound[i][3],
                             seed_bound[i][0]:seed_bound[i][0] + seed_bound[i][2]]
            cv2.imwrite(os.path.join(folder_name_cropped, imagename + '_' + str(i) + '.jpg'),
                        cv2.cvtColor(seed_segmented, cv2.COLOR_RGB2BGR))

            if display_segmented:
                print(imagename + "_" + str(i) + ".JPG")
                cv2.imshow(seed_segmented)
                print("\n")

            if not for_annotation:
                if model:  # make prediction for each seed if a model is specified
                    net, device, criterion = load_trained_model(model)
                    label, score = predict_seed(seed_segmented, model=net, device=device, criterion=criterion)
                    
                else:
                    label = None
                rows.append([imgpath, seed_bound[i][0], seed_bound[i][1], seed_bound[i][0] + seed_bound[i][2],
                             seed_bound[i][1] + seed_bound[i][3], label])
            else:  # for annotation
                if img_label == IMAGE_LABELS['GOOD']:
                    rows.append([imgpath, seed_bound[i][0], seed_bound[i][1], seed_bound[i][0] + seed_bound[i][2],
                                 seed_bound[i][1] + seed_bound[i][3], SEED_LABELS['GOOD']])
                    label = img_label
                elif img_label == IMAGE_LABELS['BAD']:
                    rows.append([imgpath, seed_bound[i][0], seed_bound[i][1], seed_bound[i][0] + seed_bound[i][2],
                                 seed_bound[i][1] + seed_bound[i][3], SEED_LABELS['BAD']])
                    label = img_label
                elif img_label == IMAGE_LABELS['UNKNOWN']:
                    # print('Cannot annotate images with no annotations')
                    # continue
                    raise Exception('Cannot annotate images with no annotations')
                elif img_label == IMAGE_LABELS['MIX']:
                    # print('Cannot annotate images with no annotations')
                    # continue
                    raise Exception('Cannot annotate images with no annotations')
                else:
                    raise Exception(
                        'Unrecognised image label! It must be one of \'GOOD\', \'BAD\', \'MIX\', \'UNKNOWN\'')
            color = get_color(label)
            cv2.rectangle(image_seed_bound,
                          (int(seed_bound[i][0] - bound_padding), int(seed_bound[i][1] - bound_padding)),
                          (int(seed_bound[i][0] + seed_bound[i][2] + bound_padding),
                           int(seed_bound[i][1] + seed_bound[i][3] + bound_padding)),
                          color, 2)
            if i == 0:
                laser_matrix[0][0] = int(label.numpy()[0])
            elif i == 1:
                laser_matrix[0][1] = int(label.numpy()[0])
            elif i == 2:
                laser_matrix[1][0] = int(label.numpy()[0])
            elif i == 3:
                laser_matrix[1][1] = int(label.numpy()[0])
            if int(label.numpy()[0]) == 1:
                good_seeds = good_seeds + 1
            elif int(label.numpy()[0]) == 0:
                bad_seeds = bad_seeds + 1
        
    tray_id = os.path.basename(imgpath)
    tray_id = os.path.splitext(tray_id)[0]
    
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y %H:%M:%S")
    
    # Write results of current tray to results.csv
    with open('out/results.csv', 'a', newline='') as file:
        file.write("\n")
        file.write(tray_id + ',' + str(good_seeds) + ',' + str(bad_seeds) + ',' + date_time)

    # Get output image with Bounding Box
    # if display_bound:
    # print(imagename + ".jpg")
    # display_resized_image(cv2.cvtColor(image_seed_bound, cv2.COLOR_RGB2BGR), IMAGE_RESIZE_SCALE_PERCENTAGE)
    folder_name_bound = "out/results"  # set output directory path
    cv2.imwrite(os.path.join(folder_name_bound, imagename + '.JPG'), cv2.cvtColor(image_seed_bound, cv2.COLOR_RGB2BGR))

    return rows, laser_matrix


def single_image_detection(input_filename, output_filename, display=True, img_label=IMAGE_LABELS['UNKNOWN'],
                           for_annotation=FOR_ANNOTATION, model=None):
    seed_mask = pre_process(input_filename, display=display)
    # seed_mask = pre_process2(input_filename, display=display)
    _, laser_matrix = segment(input_filename, seed_mask, output_filename, display_bound=display, img_label=img_label,
                for_annotation=for_annotation, model=model)
    return laser_matrix


def batch_image_detection(input_filename, output_filename, display_bound=False, img_label=None,
                          for_annotation=FOR_ANNOTATION, model=None):
    ''' perform seed detection for multiple images
    :param:
        filename: str, it can be either image file or csv file
        img_label: str, default is None. If not specified, then do detection for all the images in the record.
            Otherwise, do detection only for the image that matches the specified image label
    :return:
    '''
    df = pd.read_csv(input_filename)
    '''for i, df_row in tqdm(df.iterrows(), desc="Pre-processing and segmentation", total=df.shape[0], unit="image"):
        image_name = df_row[CSV_SEEDS_HEADER[0]]
        image_label = df_row[CSV_SEEDS_HEADER[1]]
        if (not (img_label is None)) and (img_label != image_label): # do detection for the image with the specified label
            continue
        single_image_detection(image_name, output_filename, display=display_bound, img_label=image_label,
                               for_annotation=for_annotation)'''

    with tqdm(desc="Pre-processing and segmentation", total=df.shape[0], unit="image") as pbar:
        for i, df_row in df.iterrows():
            image_name = df_row[CSV_SEEDS_HEADER[0]]
            image_label = df_row[CSV_SEEDS_HEADER[1]]
            if (not (img_label is None)) and (
                    img_label != image_label):  # do detection for the image with the specified label
                continue
            single_image_detection(image_name, output_filename, display=display_bound, img_label=image_label,
                                   for_annotation=for_annotation, model=model)
            pbar.update(1)


def run_seed_detection(input_filename):
    # read parameters from commandline if any
    args = read_from_commandline()
    for_annotation = args.annotation
    imageinput_path = args.imageinputroot
    csvinput_path = args.csvinputroot
    output_path = args.outputroot
    # input_filename = args.fileinput
    annotationoutput_filename = args.annotation2csvoutput
    detectionoutput_filename = args.detection2csvoutput
    img_label = args.imglabel
    model = [args.model_type, args.model_file]

    input_filename = r'/home/pi/Desktop/PI_SEGP/resources/temp/current_input/' + input_filename

    ############### for debug only!!!!!#############
    # input_filename = r'C:/Users/justi/Desktop/AAR/csv/seeds_normallight.csv'
    # input_filename = r'C:/Users/justi/Desktop/AAR/dataset/seeds_batch_1/test/GoodSeed/GoodSeed1.jpg'
    # input_filename = r'C:/Users/justi/Desktop/AAR/dataset/seeds_batch_1/train/BadSeed/BadSeed102.jpg'
    # input_filename = r'C:/Users/justi/Desktop/AAR/dataset/NormalRoomLighting/Set15/Line_Bad_Seeds (s15).jpg'
    # input_filename = r'C:/Users/justi/Desktop/AAR/dataset/LightBox/Set20/SpaceOutRandom_Mix (s20).jpg'
    # for_annotation = False
    ######### end of debugging code#################

    # detect invididual seeds for the input image(s) and store the output in csv files either as annotations or detection
    # determine if the input file is image file or csv file
    # If it is a single image file, then the  mode has to be not for_annotation
    # If it is a csv file, we can determine the mode based on default info or overriding info from the commandline
    extension = input_filename.split('.')[1]
    # check if the input file exists
    if not os.path.exists(input_filename):
        # join the input_filename with imageinput_path
        input_filename = os.path.join(imageinput_path, input_filename)
    if not os.path.exists(input_filename):
        raise Exception('File {0} does not exist!'.format(input_filename))

    if for_annotation:
        outcsv = create_empty_annotation_csv(output_path, annotationoutput_filename)
    else:
        outcsv = create_empty_detection_csv(output_path, detectionoutput_filename)

    if extension == IMAGE_FORMAT:
        # call single image detection function
        img_label, _ = check_img_label(input_filename)
        laser_matrix = single_image_detection(input_filename, outcsv, img_label=img_label, for_annotation=for_annotation, model=model)
    elif extension == 'csv':
        # call seed detection reading from csv file
        batch_image_detection(input_filename, outcsv, img_label=img_label, for_annotation=for_annotation, model=model)
    else:
        raise Exception('Expecting the input file to be either {0} or CSV!'.format(IMAGE_FORMAT))
# def startProgram(input_filename):
#     for_annotation = args.annotation
#     imageinput_path = args.imageinputroot
#     csvinput_path = args.csvinputroot
#     output_path = args.outputroot
#     input_filename = args.fileinput
#     annotationoutput_filename = args.annotation2csvoutput
#     detectionoutput_filename = args.detection2csvoutput
#     img_label = args.imglabel
#     model = [args.model_type, args.model_file]
#
#     outcsv = create_empty_annotation_csv(output_path, annotationoutput_filename)
#     input_filename = os.path.join(imageinput_path, input_filename)
#     img_label, _ = check_img_label(input_filename)
#     single_image_detection(input_filename, outcsv, "UNKNOWN", for_annotation, model=model)
    return laser_matrix

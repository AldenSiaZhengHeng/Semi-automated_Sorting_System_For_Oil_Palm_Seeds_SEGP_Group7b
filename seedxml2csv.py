'''
# This file is provided by our supervisor, Dr Iman Yi Liao.
# Given a path, the function is to look through all
# the subfolders to convert xml annotation files to
# csv files for oil palm seeds detection and classification
#
# The function is written for the data collection in the
# second batch, where the folder structure is as follows.
# ..\
# ......\Set1
# ..........SpaceOutRandom_Mix (s1).JPG
# ..........SpaceOutRandom_Mix (s1).xml
# ..........SpaceOutRandom_Good_Seeds (s1).JPG
# ..........SpaceOutRandom_Bad_Seeds (s1).JPG
# ..........Line_Mix (s1).JPG
# ..........Line_Mix (s1).xml
# ..........Line_Good_Seeds (s1).JPG
# ..........Line_Bad_Seeds (s1).JPG
# ......\Set2
# ......
#
# by Dr. Iman Yi Liao, 22 June 2021
'''

# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd
import os
import glob
import argparse
from utils.const import *
from utils.seed_io import read_from_commandline
from create_csv_files import create_empty_annotation_csv


# Define the column header in the csv file for individual oil palm seeds
# bbox_annotate stores value 'YES' if the bounding box for each individual seed in the image is annotated, otherwise 'NO'
# bbox_label stores value 'GOOD' if it's annotated as good seed, otherwise 'BAD'.
cols = CSV_SEED_INDIVIDUAL_HEADER
# Define the default csv file name that stores the individual seed information
outcsv = os.path.join(CODE_ROOT, 'csv', CSV_SEED_INDIVIDUAL_ANNOTATED)


# Define function to convert xml file annotation of an input image of a set of seeds
def xml2csv(file_name):
    #folder_name and file_name are to be passed as known parameters
    rows = []
    # Parsing the XML file
    xmlparse = Xet.parse(file_name)
    root = xmlparse.getroot()
    for item in root.findall('object'):
        label = SEED_LABELS['GOOD'] if item.find('name').text == 'Good Seed' else SEED_LABELS['BAD']
        pos = item.find('bndbox')
        x_min = int(pos.find('xmin').text)
        y_min = int(pos.find('ymin').text)
        x_max = int(pos.find('xmax').text)
        y_max = int(pos.find('ymax').text)

        # we need to store the name of the jpg file instead of xml file
        rows.append({CSV_SEED_INDIVIDUAL_HEADER[0]: file_name.split('.')[0] + '.jpg',
                     CSV_SEED_INDIVIDUAL_HEADER[1]: x_min,
                     CSV_SEED_INDIVIDUAL_HEADER[2]: y_min,
                     CSV_SEED_INDIVIDUAL_HEADER[3]: x_max,
                     CSV_SEED_INDIVIDUAL_HEADER[4]: y_max,
                     CSV_SEED_INDIVIDUAL_HEADER[5]: label})

    return rows


# Define function to loop through all sub-folders of a given directory to find all
# xml files and return the path and the xml file name
def findXMLfiles(root):
    # use glob module for path search including sub-folders
    path = os.path.join(root, '**/*.xml')
    file_names = []
    for filename in glob.iglob(path, recursive=True):
        file_names.append(filename)
    return file_names

'''
# Define function to parse the arguments of command line
def getInputDir():
    # Construct a parser
    parser = argparse.ArgumentParser()
    # There should be at least one argument of the input directory
    # There could be more than one directories
    parser.add_argument('paras', type=str, nargs='+')
    # There may be an output file name
    parser.add_argument('-o', '--out', nargs='?', const=outcsv, default=outcsv)
    args = parser.parse_args()

    return args.paras, args.out
'''

def seedxml2csv(input_path, output_path, output_file_name):
    '''Get all input directories
    In case there are more than one directories, we need to find xml files under all those directories
    and convert them in one csv file
    '''
    #input_roots, outfile_name = getInputDir()

    # For xml files under all directories and convert them into a single csv file
    rows = []
    file_names = findXMLfiles(input_path)
    for j, filename in enumerate(file_names):
        current_rows = xml2csv(filename)
        rows += current_rows
    df = pd.DataFrame(rows, columns=cols)

    # Writing dataframe to csv
    outfile_name = create_empty_annotation_csv(output_path, output_file_name)
    # append to the existing file or the file that was just created above
    df.to_csv(outfile_name, mode='a', header=False, index=False)
  
      
if __name__ == "__main__":
    # read from commandline
    args = read_from_commandline()
    input_path = args.imageinputroot
    output_path = args.outputroot
    annotation2csv_filename = args.annotation2csvoutput

    # retrieve all the annotations
    seedxml2csv(input_path, output_path, annotation2csv_filename)

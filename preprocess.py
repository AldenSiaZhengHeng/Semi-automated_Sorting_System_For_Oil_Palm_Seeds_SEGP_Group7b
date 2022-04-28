'''
# This file is provided by our supervisor, Dr Iman Yi Liao.
# This file is used to preprocess the image before performing seeds classification
'''

import cv2
import numpy as np
from utils.const import *
from utils.seed_io import display_resized_image


def get_mask_withRGB(image):
    # Otsu's thresholding method
    ret2, mask = cv2.threshold(image[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert mask
    mask = cv2.bitwise_not(mask)

    # make mask bigger
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel)

    # remove holes in seed
    kernel = np.ones((24, 24), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # clear small dots
    kernel = np.ones((64, 64), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def pre_process2(path, display=False):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    mask = get_mask_withRGB(image)
    # Display seed mask
    if display:
        print(path.rsplit('\\', 2)[2])
        display_resized_image(mask, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title='Seed mask')
        print("\n")

    return mask


def get_mask(image, filename=None):
    """
    Function for image pre-processing
    Parameters:
        path : str, Image path.
        display : bool, default=False, Display seed mask.

    Returns:
        seed_mask : ndarray, Seed mask for image segmentation.
    """
    kernel15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    #kernel21 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Get seed part 1
    # Otsu's thresholding on saturation channel
    _, seed_part_1_otsu = cv2.threshold(image_hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to get main seed parts
    seed_part_1_closing = cv2.morphologyEx(seed_part_1_otsu, cv2.MORPH_CLOSE, kernel15)

    # Get seed part 2
    # Morphological dilation on value channel
    seed_part_2_dilated = cv2.dilate(image_hsv[:,:,2], kernel15)

    # Gaussian blur to smoothen image
    seed_part_2_gaussian = cv2.GaussianBlur(seed_part_2_dilated, (15, 15), 0)

    # Compute difference to get edges
    seed_part_2_diff = cv2.absdiff(image_hsv[:,:,2], seed_part_2_gaussian)

    # Otsu's thresholding to get clear edges
    _, seed_part_2_otsu = cv2.threshold(seed_part_2_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get seed part 3
    # Canny edge detection
    seed_part_3_canny = cv2.Canny(image_hsv[:,:,1], 100, 200)

    # Morphological closing to get main seed parts
    seed_part_3_closing = cv2.morphologyEx(seed_part_3_canny, cv2.MORPH_CLOSE, kernel15)

    # Combine seed parts
    seed_combined = np.clip(seed_part_1_otsu + seed_part_1_closing + seed_part_2_otsu + seed_part_3_closing, 0, 255)

    # Morphological closing to get main seed parts
    seed_closing = cv2.morphologyEx(seed_combined, cv2.MORPH_CLOSE, kernel15)
    #seed_closing = seed_combined

    # Fill holes from point (0, 0)
    mask_fill_holes = np.zeros((seed_closing.shape[0] + 2, seed_closing.shape[1] + 2), np.uint8)
    seed_fill = seed_closing.copy()
    cv2.floodFill(seed_fill, mask_fill_holes, (0, 0), 255)

    # Invert floodfilled image
    seed_fill_inv = ~seed_fill

    # Combine the two images to get the foreground.
    seed_mask = seed_closing | seed_fill_inv
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel15)
    #seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel21)

    seed_part_1_closing = np.expand_dims(seed_part_1_closing, axis=2)
    seed_part_2_otsu = np.expand_dims(seed_part_2_otsu, axis=2)
    seed_part_3_closing = np.expand_dims(seed_part_3_closing, axis=2)
    seed_mask_channels = np.concatenate((seed_part_1_closing, seed_part_2_otsu, seed_part_3_closing), axis=2)

    return seed_mask, seed_mask_channels
    #return seed_part_2_otsu, seed_mask_channels

'''
def get_mask(image, filename=None):
    # obtain mask from RGB image
    # mask_RGB = get_mask_withRGB(image)

    kernel15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel21 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #########for debug only############
    if filename is None:
        filename = ''
    display_resized_image(image_hsv[:,:,0], IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'Hue' + '.jpg', save2file=True)
    ###################################

    #########for debug only############
    display_resized_image(image_hsv[:,:,1], IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'Saturation' + '.jpg', save2file=True)
    ###################################

    #########for debug only############
    display_resized_image(image_hsv[:,:,2], IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'Value' + '.jpg', save2file=True)
    ###################################

    # Get seed part 1
    # Otsu's thresholding on saturation channel
    _, seed_part_1_otsu = cv2.threshold(image_hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #########for debug only############
    display_resized_image(seed_part_1_otsu, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_1_otsu' + '.jpg', save2file=True)
    ###################################

    # Morphological closing to get main seed parts
    seed_part_1_closing = cv2.morphologyEx(seed_part_1_otsu, cv2.MORPH_CLOSE, kernel15)

    #########for debug only############
    display_resized_image(seed_part_1_closing, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_1_closing' + '.jpg', save2file=True)
    ###################################

    # Get seed part 2
    # Morphological dilation on value channel
    seed_part_2_dilated = cv2.dilate(image_hsv[:,:,2], kernel15)
    #########for debug only############
    display_resized_image(seed_part_2_dilated, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_2_dilated' + '.jpg', save2file=True)
    ###################################

    # Gaussian blur to smoothen image
    seed_part_2_gaussian = cv2.GaussianBlur(seed_part_2_dilated, (15, 15), 0)
    #########for debug only############
    display_resized_image(seed_part_2_gaussian, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_2_gaussian' + '.jpg', save2file=True)
    ###################################

    # Compute difference to get edges
    seed_part_2_diff = cv2.absdiff(image_hsv[:,:,2], seed_part_2_gaussian)
    #########for debug only############
    display_resized_image(seed_part_2_diff, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_2_diff' + '.jpg', save2file=True)
    ###################################

    # Otsu's thresholding to get clear edges
    _, seed_part_2_otsu = cv2.threshold(seed_part_2_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #########for debug only############
    display_resized_image(seed_part_2_otsu, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_2_otsu' + '.jpg', save2file=True)
    ###################################

    # Get seed part 3
    # Canny edge detection
    seed_part_3_canny = cv2.Canny(image_hsv[:,:,1], 100, 200)
    #########for debug only############
    display_resized_image(seed_part_3_canny, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_3_canny' + '.jpg', save2file=True)
    ###################################

    # Morphological closing to get main seed parts
    seed_part_3_closing = cv2.morphologyEx(seed_part_3_canny, cv2.MORPH_CLOSE, kernel15)
    #########for debug only############
    display_resized_image(seed_part_3_closing, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_part_3_closing' + '.jpg', save2file=True)
    ###################################

    # Combine seed parts
    seed_combined = np.clip(seed_part_1_otsu + seed_part_1_closing + seed_part_2_otsu + seed_part_3_closing, 0, 255)
    #########for debug only############
    display_resized_image(seed_combined, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_combined' + '.jpg', save2file=True)
    ###################################

    # Morphological closing to get main seed parts
    seed_closing = cv2.morphologyEx(seed_combined, cv2.MORPH_CLOSE, kernel15)
    #seed_closing = seed_combined
    #########for debug only############
    display_resized_image(seed_closing, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_'+ 'seed_closing' + '.jpg', save2file=True)
    ###################################

    # Fill holes from point (0, 0)
    mask_fill_holes = np.zeros((seed_closing.shape[0] + 2, seed_closing.shape[1] + 2), np.uint8)
    seed_fill = seed_closing.copy()
    cv2.floodFill(seed_fill, mask_fill_holes, (0, 0), 255)
    #########for debug only############
    display_resized_image(seed_fill, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_fill' + '.jpg', save2file=True)
    ###################################

    # Invert floodfilled image
    seed_fill_inv = ~seed_fill
    #########for debug only############
    display_resized_image(seed_fill_inv, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_fill_inv' + '.jpg', save2file=True)
    ###################################

    # Combine the two images to get the foreground.
    #seed_mask = (seed_closing | seed_fill_inv) | mask_RGB
    seed_mask = (seed_closing | seed_fill_inv)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel15)
    #seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel21)
    #########for debug only############
    display_resized_image(seed_mask, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title=filename + '_' + 'seed_mask' + '.jpg', save2file=True)
    ###################################

    seed_part_1_closing = np.expand_dims(seed_part_1_closing, axis=2)
    seed_part_2_otsu = np.expand_dims(seed_part_2_otsu, axis=2)
    seed_part_3_closing = np.expand_dims(seed_part_3_closing, axis=2)
    seed_mask_channels = np.concatenate((seed_part_1_closing, seed_part_2_otsu, seed_part_3_closing), axis=2)

    return seed_mask, seed_mask_channels
'''

def pre_process(path, display=False):
    filename = path.rsplit('/', 1)[1].split('.')[0]
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    seed_mask, _ = get_mask(image, filename)
    # Display seed mask
    '''
    if display:
        print(path.rsplit('/', 2)[2])
        display_resized_image(seed_mask, IMAGE_RESIZE_SCALE_PERCENTAGE, img_title='_' + 'Seed mask')
        print("\n")
    '''
    return seed_mask


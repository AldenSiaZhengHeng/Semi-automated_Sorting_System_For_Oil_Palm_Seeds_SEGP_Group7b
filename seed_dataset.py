'''
# This file is provided by our supervisor, Dr Iman Yi Liao.
# This file is to contain the function to read the dataset for training model provided.
'''


import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from torch.utils.data import DataLoader
import csv
import cv2
from utils.const import *
from preprocess import get_mask


# Define customised transforms
class ToHSV(object):
    """
    Convert RGB to HSV
    """
    def __init__(self):
        pass

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # convert back to np.array
            new_img = image.permute(1,2,0).numpy()
        #print(new_img.shape)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
        new_img = transforms.ToTensor()(new_img).float()
        return new_img


class CatHSV(object):
    """
    Concatenate HSV to RGB
    """
    def __init__(self):
        pass

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # convert back to np.array
            image = image.permute(1,2,0).numpy()
        new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        new_img = np.concatenate((image, new_img), 2)   # concatenate along channel dimension (H, W, C)
        new_img = transforms.ToTensor()(new_img).float()
        return new_img


class CatMask(object):
    """
    Concatenate the masks of the seed with the image
    """
    def __init__(self):
        pass

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # convert back to np.array
            image = image.permute(1,2,0).numpy()
            image = np.uint8(image)
        new_img, _ = get_mask(image)
        new_img = np.expand_dims(new_img, axis=2)
        new_img = np.concatenate((image, new_img), 2)   # concatenate along channel dimension (H, W, C)
        new_img = transforms.ToTensor()(new_img).float()
        return new_img


class CatMaskChannels(object):
    """
    Concatenate the masks of the seed with the image
    """
    def __init__(self):
        pass

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # convert back to np.array
            image = image.permute(1,2,0).numpy()
            image = np.uint8(image)
        _, new_img = get_mask(image)
        new_img = np.concatenate((image, new_img), 2)   # concatenate along channel dimension (H, W, C)
        new_img = transforms.ToTensor()(new_img).float()
        return new_img


# Define transformations for training set
train_transforms = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_for_torchvision_models = transforms.Compose([
    #transforms.RandomResizedCrop(MODEL_INPUT_SIZE),
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms_for_torchvision_inception = transforms.Compose([
    #transforms.RandomResizedCrop(MODEL_INPUT_SIZE),
    transforms.Resize([299, 299]),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms_HSV = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    ToHSV(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_catHSV = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatHSV(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_catMask = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatMask(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transforms_catMaskChannels = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatMaskChannels(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    #transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transformations for testing set
test_transforms = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms_for_torchvision_models = transforms.Compose([
    transforms.Resize([224, 224]),
    #transforms.CenterCrop(MODEL_INPUT_SIZE),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms_for_torchvision_inception = transforms.Compose([
    transforms.Resize([299, 299]),
    #transforms.CenterCrop(MODEL_INPUT_SIZE),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.1307,), (0.3081,))
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_HSV = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    ToHSV(),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_catHSV = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatHSV(),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_catMask = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatMask(),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform_catMaskChannels = transforms.Compose([
    transforms.Resize([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    CatMaskChannels(),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize((0.1307,), (0.3081,))
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Generate dataset from csv file
def generate_dataset(csvfile, with_test=True):
    df = pd.read_csv(csvfile)
    if with_test:
        train_val, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])
        train, validate = np.split(train_val.sample(frac=1, random_state=42), [int(.8*len(df))])
    else:
        train, validate = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df))])
        test = None
    #print(train.shape, validate.shape)
    #print(validate)
    return train, validate, test


# Load image data from dataframe, (image_path, label, annotation_file_path)
def image_dataloader(df):
    # load data from Pandas' DataFrame structure
    dataset = []
    for i in range(len(df)):
        current = df.iloc[i]
        image = cv2.imread(current[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        # print(image.shape)
        image = cv2.resize(image, (int(image.shape[1]*IMAGE_RESIZE_SCALE_PERCENTAGE/100),
                                   int(image.shape[0]*IMAGE_RESIZE_SCALE_PERCENTAGE/100))) # I'm resizing the image but you should try to do some appropriate pre-processing to obtaining smaller image size
        dataset.append((image, current[1], current[2]))
        print('Data processed', i)

    return dataset


class IndividualSeedDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        # it can read either from annotated file with bounding box coordinates on an image of multiple seeds or from csv
        # file of cropped seed images
        filenames = item[0].rsplit('.')
        if not os.path.exists(item[0]):
            # check uppercase JPG or lowercase jpg
            filename = filenames[0] + '.' + filenames[1].upper()
            if not os.path.exists(filename):
                filename = filenames[0] + '.' + filenames[1].lower()
                if not os.path.exists(filename):
                    raise Exception('File does not exist!')
        else:
            filename = item[0]
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        if len(item) > 2: # read from bounding box
            label = item[5]
            cropped_image = image[item[2]:item[4], item[1]:item[3]]  # y_min:y_max, x_min:x_max,
        else: # read from cropped seed image directly
            label = item[1]
            cropped_image = image

        if label == 'GOOD':
            label = 1
        elif label == 'BAD':
            label = 0
        else:
            raise Exception("Unrecognised seed label - choose from either 'GOOD' or 'BAD'")
        if not os.path.exists(filename):
            raise Exception('The seed image does not exist!')
        # normalise the cropped seed image between 0 and 1
        #cropped_image = cv2.normalize(cropped_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cropped_image = transforms.ToTensor()(cropped_image).float()
        #### for debug only #####
        #print(image.shape)
        #print(idx, ': \t', cropped_image.shape, '\t', item[2], '\t', item[4])
        #########################

        # if any transformation is needed, e.g., to resize the image
        if self.transform:
            cropped_image = self.transform(cropped_image)
        if self.target_transform:
            label = self.target_transform(label)

        return cropped_image, label


# Define train transforms for SeedDetectionDataset
train_transforms_for_detection = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5)
])

# convert csv record of annotations of individual seeds in an image to the format that fits
# torchvision.models.detection.fasterrcnn
def csv_to_lists(df):
    # find the unique set of image paths
    list_image_paths = df.file_name.unique()
    # For each unique image path, find the set of boxes and labels
    list_targets = []
    for path in list_image_paths:
        rows = df.loc[df['file_name'] == path]
        rows = rows.to_numpy()
        list_targets.append({'boxes': rows[:, 1:5], 'labels': rows[:, 5:6]})
    # return the list of image paths and the list of targets
    return list_image_paths, list_targets


# Define seed detection dataset
class SeedDetectionDataset(Dataset):
    """
    The input is a csv file that contains the full path of the image, the box coordinate x_min, y_min, x_max, y_max, and
    the label of the box. The dataset should return a tuple of (image, target) where the image is the image in tensor
    type, and the target contains a dictionary of {k:v} where k is in {'boxes', 'labels'} and v for 'boxes' is a float
    tensor of shape Nx4, and v for 'labels' is an int64 tensor for shape N
    """
    def __init__(self, df, transform=None, target_transform=None):
        # restructure self.df to a list of fullpath of images, a list of dictionaries where each dictionary contains k:v
        # such that {'boxes': FloatTensor Nx4, 'labels': Int64Tensor N}
        self.list_image_paths, self.list_targets = csv_to_lists(df)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.list_image_paths)

    def __getitem__(self, idx):
        image_path = self.list_image_paths[idx]
        if not os.path.exists(image_path):
            raise Exception('The seed image does not exist!')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # normalise the cropped seed image between 0 and 1
        image = transforms.ToTensor()(image).float()

        target = self.list_targets[idx]
        # convert labels from string to integer
        labels = target['labels']
        labels[labels == 'GOOD'] = 1
        labels[labels == 'BAD'] = 0
        labels = labels.astype(int)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        boxes = target['boxes']
        boxes = boxes.astype(float)
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)

        # if any transformation is needed, e.g., to resize the image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


'''
# load the training csv file in terms of annotations to dataframe and randomly split it to training and validation sets respectively
trainvaldf = pd.read_csv("trainingdata.csv")
traindf, valdf = np.split(trainvaldf.sample(frac=1, random_state=42), [int(.8*len(trainvaldf))])
base_path = '/content/drive/My Drive/AAR/seed/'

# Create training and validation dataset with OilPalmSeedsDataset
train_dataset = IndividualSeedDataset(traindf, transform=transform)
val_dataset = IndividualSeedDataset(valdf, transform=transform)

print('training set', len(train_dataset))
print('val set', len(val_dataset))

# load the testing csv file as dataframe
testdf = pd.read_csv("testdata.csv")
test_dataset = IndividualSeedDataset(testdf, transform=transforms.Resize([224, 224]))
print('test set', len(test_dataset))

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)

for i, sample in enumerate(train_dataloader):
    if i > 0:
        break
    print(sample['image'].size())
    print(sample['label'].size())
    plt.imshow(sample['image'][0].permute(1,2,0)) # change (C,H,W) to (H,W,C)
    plt.show()
    print(f"Label: {sample['label'][0]}")
'''

'''
# This file is provided by our supervisor, Dr Iman Yi Liao.
# There are a little bit modification had done in this file to match to our system
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import copy
import random
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics
import time
from utils.const import *
from seed_dataset import *
from src.base_networks import CustomUnetGenerator
from src.cbam import CBAM
from preprocess import get_mask
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

WORKING_DIR = os.getcwd()


# CNN model proposed by group 1
class NetG1(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256)):
        super(NetG1, self).__init__()
        self.input_img_size = input_img_size  # the input size of a single image (C,H,W)
        self.conv1 = nn.Conv2d(self.input_img_size[0], 6, 5, 3)
        self.conv2 = nn.Conv2d(6, 12, 5, 3)
        self.conv3 = nn.Conv2d(12, 24, 5, 3)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        # The input number to the first fully connected layer is a bit tricky
        # We can run through a dummy forward pass to obtain the input number to the first fully connected layer
        dummydata = torch.rand(1, *self.input_img_size)
        dummydata = F.leaky_relu(self.conv1(dummydata))
        dummydata = F.leaky_relu(self.conv2(dummydata))
        dummydata = F.leaky_relu(self.conv3(dummydata))
        self.num_features_2fc = dummydata.shape[0] * dummydata.shape[1] * dummydata.shape[2] * dummydata.shape[
            3]  # there are clever ways but I don't recall them and need to search up
        # print(self.num_features_2fc)  # for debugging
        self.fc1 = nn.Linear(self.num_features_2fc, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


# CNN model proposed by group 2
# Dominant Color Descriptor
def dcd(image, image_size, k_value):
    y = []
    for index in range(image.size(0)):
        img = image[index, :, :, :].permute(1, 2, 0).cpu().numpy()

        img = cv2.resize(img, (image_size, image_size))

        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, labels, center = cv2.kmeans(Z, k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[labels.flatten()]
        res2 = res.reshape((img.shape))

        # convert to hsv
        hsv = cv2.cvtColor(res2, cv2.COLOR_RGB2HSV)

        # normalize to between 0 and 1
        hsv = cv2.normalize(hsv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        y.append(hsv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.FloatTensor(y).to(device)
    y = y.permute(0, 3, 1, 2)

    return y


class NetG2(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), use_colour_descriptor=True):
        super().__init__()
        self.input_img_size = input_img_size
        self.use_colour_descriptor = use_colour_descriptor
        self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        # initialize linear layer inputs
        x = torch.randn(1, *self.input_img_size)
        self._to_linear1 = None
        if self.use_colour_descriptor:
            self._to_linear2 = None
            self.dcds(x)
        self.convs(x)

        if self.use_colour_descriptor:
            if self._to_linear1 is None or self._to_linear2 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1 + self._to_linear2
        else:
            if self._to_linear1 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1

        self.fc1 = nn.Linear(inputLinear, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))

        # Flatten function
        if self._to_linear1 is None:
            self._to_linear1 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def dcds(self, x):
        x = dcd(x, 32, 6)

        # Flatten function
        if self._to_linear2 is None:
            self._to_linear2 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        if self.use_colour_descriptor:
            y = self.dcds(x)
            y = y.contiguous().view(-1, self._to_linear2)
        x = self.convs(x)
        x = x.contiguous().view(-1, self._to_linear1)
        if self.use_colour_descriptor:
            x = torch.cat((x, y), 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# G2 with Attention Unit
class NetG2_AU(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), mask_type = 'edge', use_colour_descriptor=True):
        super(NetG2_AU, self).__init__()
        self.input_img_size = input_img_size
        self.mask_type = mask_type
        self.use_colour_descriptor = use_colour_descriptor
        self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        # initialize linear layer inputs
        x = torch.randn(1, *self.input_img_size)
        self._to_linear1 = None
        if self.use_colour_descriptor:
            self._to_linear2 = None
            self.dcds(x)

        # create Attention Unit
        connection_size = self.convs(x).shape[2:]
        self.downsample = nn.Upsample(connection_size, mode='bilinear', align_corners=False)
        self.attention_unit = CustomUnetGenerator(256 + 1, 256, num_downs=0, ngf=32, last_act='tanh')

        if self.use_colour_descriptor:
            if self._to_linear1 is None or self._to_linear2 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1 + self._to_linear2
        else:
            if self._to_linear1 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1

        self.fc1 = nn.Linear(inputLinear, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))

        # Flatten function
        if self._to_linear1 is None:
            self._to_linear1 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def dcds(self, x):
        x = dcd(x, 32, 6)

        # Flatten function
        if self._to_linear2 is None:
            self._to_linear2 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, imgs):
        masks = []
        for index in range(imgs.size(0)):
            img = imgs[index, :, :, :].permute(1, 2, 0).cpu().numpy()
            if self.mask_type == 'combined':
                mask, _ = get_mask(np.uint8(img))
            elif self.mask_type == 'edge':
                _, mask = get_mask(np.uint8(img))
                mask = mask[:,:,1]
            elif self.mask_type == 'sprout':
                _, mask = get_mask(np.uint8(img))
                mask = mask[:,:,0]
            elif self.mask_type == 'sprout+edge':
                _, mask = get_mask(np.uint8(img))
                mask = mask[:,:,0] | mask[:,:,1]
            else:
                raise Exception('Mask type is not defined! Choose from [\'combined\', \'edge\', \'sprout\', \'sprout+edge\']')
            # normalize to between 0 and 1
            mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            mask = np.expand_dims(mask, 2) # expand from (H,W) to (H,W,C)
            masks.append(mask)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        masks = torch.FloatTensor(masks).to(device)
        masks = masks.permute(0, 3, 1, 2)

        if self.use_colour_descriptor:
            y = self.dcds(imgs)
            y = y.contiguous().view(-1, self._to_linear2)
        out = self.convs(imgs)

        # go through attention map with mask as input
        out2 = self.downsample(masks)
        attention_map = torch.cat([out2, out], dim=1)
        attention_map = self.attention_unit(attention_map)
        # new_conv_feature = (1 + attention_map) * out
        out = (1 + attention_map) * out

        out = out.contiguous().view(-1, self._to_linear1)
        if self.use_colour_descriptor:
            out = torch.cat((out, y), 1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# G2 with CBAM (Convolutional Block Attention Module)
class NetG2_CBAM(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), mask_type='cbam', use_colour_descriptor=True):
        super(NetG2_CBAM, self).__init__()
        self.input_img_size = input_img_size
        self.mask_type = mask_type
        self.use_colour_descriptor = use_colour_descriptor
        self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
        #self.cbam1 = CBAM(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        #self.cbam2 = CBAM(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.cbam3 = CBAM(256)

        # initialize linear layer inputs
        x = torch.randn(1, *self.input_img_size)
        self._to_linear1 = None
        if self.use_colour_descriptor:
            self._to_linear2 = None
            self.dcds(x)

        # create Attention Unit
        connection_size = self.convs(x).shape[2:]
        self.downsample = nn.Upsample(connection_size, mode='bilinear', align_corners=False)
        self.attention_unit = CustomUnetGenerator(256 + 1, 256, num_downs=0, ngf=32, last_act='tanh')

        if self.use_colour_descriptor:
            if self._to_linear1 is None or self._to_linear2 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1 + self._to_linear2
        else:
            if self._to_linear1 is None:
                inputLinear = None
            else:
                inputLinear = self._to_linear1

        self.fc1 = nn.Linear(inputLinear, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
        x = x + self.cbam3(x)

        # Flatten function
        if self._to_linear1 is None:
            self._to_linear1 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def dcds(self, x):
        x = dcd(x, 32, 6)

        # Flatten function
        if self._to_linear2 is None:
            self._to_linear2 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, imgs):
        if self.use_colour_descriptor:
            y = self.dcds(imgs)
            y = y.contiguous().view(-1, self._to_linear2)

        out = self.convs(imgs)

        out = out.contiguous().view(-1, self._to_linear1)
        if self.use_colour_descriptor:
            out = torch.cat((out, y), 1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# G2 with Class Token
class NetG2_ClassToken(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), token_level=0, token_channel=1, mask_type='conv', use_colour_descriptor=True):
        super(NetG2_ClassToken, self).__init__()
        self.input_img_size = input_img_size
        self.mask_type = mask_type
        self.use_colour_descriptor = use_colour_descriptor
        self.token_level = token_level
        self.token_channel = token_channel
        # define class token
        if self.token_level == 0: #at input level
            W = self.input_img_size[1]
            H = self.input_img_size[2]
        elif self.token_level == 1: # at first conv layer, same level as the feature maps of the 1st conv layer
            x = torch.randn(1, *self.input_img_size)
            x = F.max_pool2d(nn.Conv2d(self.input_img_size[0],1,7)(x), (4, 4))
            W = x.shape[2]
            H = x.shape[3]
        elif self.token_level == 2: # at the 2nd conv layer, the same level as the feature maps of the 2nd conv layer
            x = torch.randn(1, *self.input_img_size)
            x = F.max_pool2d(nn.Conv2d(self.input_img_size[0], 1, 7)(x), (4, 4))
            x = F.max_pool2d(nn.Conv2d(1, 1, 3)(x), (4, 4))
            W = x.shape[2]
            H = x.shape[3]
        else:
            raise Exception('Class token has to be in either level 0, 1, or 2')
        self.cls_token = nn.Parameter(torch.zeros(1, self.token_channel, W, H))
        trunc_normal_(self.cls_token, std=.02)
        if self.mask_type.lower() == 'conv':
            if self.token_level == 0:
                self.conv1 = nn.Conv2d(self.input_img_size[0]+self.cls_token.shape[1], 64, 7)
                self.conv2 = nn.Conv2d(64, 128, 3)
                self.conv3 = nn.Conv2d(128, 256, 3)
            elif self.token_level == 1:
                self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
                self.conv2 = nn.Conv2d(64 + self.cls_token.shape[1], 128, 3)
                self.conv3 = nn.Conv2d(128, 256, 3)
            elif self.token_level == 2:
                self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
                self.conv2 = nn.Conv2d(64, 128, 3)
                self.conv3 = nn.Conv2d(128 + self.cls_token.shape[1], 256, 3)
            else:
                raise Exception('Class token has to be in either level 0, 1, or 2')
        elif self.mask_type.lower() == 'outer':
            self.conv1 = nn.Conv2d(self.input_img_size[0], 64, 7)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.conv3 = nn.Conv2d(128, 256, 3)
        else:
            raise Exception('The type of class token can either be \'conv\' or \'outer\'')

        # initialize linear layer inputs
        x = torch.randn(1, *self.input_img_size)
        self._to_linear1 = None
        _ = self.convs(x)  # just to initialise the linear layer

        if self._to_linear1 is None:
            inputLinear = None
        else:
            inputLinear = self._to_linear1

        self.fc1 = nn.Linear(inputLinear, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def convs(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1, -1)
        if self.mask_type == 'conv':
            if self.token_level == 0:
                x = torch.cat((cls_tokens, x), dim=1)
                x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
                x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
                x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
            elif self.token_level == 1:
                x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
                x = torch.cat((cls_tokens, x), dim=1)
                x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
                x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
            elif self.token_level == 2:
                x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
                x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
                x = torch.cat((cls_tokens, x), dim=1)
                x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
            else:
                raise Exception('Class token has to be in either level 0, 1, or 2')
        elif self.mask_type == 'outer':
            x = torch.einsum('bcmn, bkij -> bkmj', cls_tokens, x)
            x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
            x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
            x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
        else:
            raise Exception('The type of class token can either be \'conv\' or \'outer\'')

        # Flatten function
        if self._to_linear1 is None:
            self._to_linear1 = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            # only take the channel that corresponds to cls_token as the input to the linear layer
            #self._to_linear1 = x[0].shape[1] * x[0].shape[2]
        #return x[:, 0:1, :, :]
        return x

    def forward(self, imgs):
        out = self.convs(imgs)

        out = out.contiguous().view(-1, self._to_linear1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# G2 with Multi-Path
class BasicPath(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), conv_channel=16):
        super(BasicPath, self).__init__()
        self.conv_channel = conv_channel
        self.input_img_size = input_img_size
        self.conv1 = nn.Conv2d(self.input_img_size[0], self.conv_channel, 7)
        self.conv2 = nn.Conv2d(self.conv_channel, self.conv_channel*2, 3)
        self.conv3 = nn.Conv2d(self.conv_channel*2, self.conv_channel*4, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv2(x), (4, 4)))
        x = F.relu(F.max_pool2d(self.conv3(x), (4, 4)))
        return x


class NetG2_MultiPath(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256), npath=4, conv_channel=16,
                 mask_type='classtoken', use_colour_descriptor=True):
        super(NetG2_MultiPath, self).__init__()
        self.input_img_size = input_img_size
        self.mask_type = mask_type
        self.use_colour_descriptor = use_colour_descriptor
        self.number_path = npath
        self.conv_channel = conv_channel
        self.convpath = BasicPath(self.input_img_size, self.conv_channel)

        # initialize linear layer inputs
        x = torch.randn(1, *self.input_img_size)
        self._to_linear1 = None
        _ = self.convs(x) # just to initialise the linear layer

        if self._to_linear1 is None:
            inputLinear = None
        else:
            inputLinear = self._to_linear1

        self.fc1 = nn.Linear(inputLinear, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def convs(self, x):
        out = self.convpath(x)
        for i in range(self.number_path-1):
            out = torch.cat((out, self.convpath(x)), dim=1)

        # Flatten function
        if self._to_linear1 is None:
            self._to_linear1 = out[0].shape[0] * out[0].shape[1] * out[0].shape[2]
        return out

    def forward(self, imgs):
        out = self.convs(imgs)

        out = out.contiguous().view(-1, self._to_linear1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# CNN model proposed by group 7
class NetG7(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256)):
        super(NetG7, self).__init__()
        self.input_img_size = input_img_size
        self.conv1 = nn.Conv2d(self.input_img_size[0], 32, 3, padding=1)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout2d(0.4)
        # The input number to the first fully connected layer is a bit tricky
        # We can run through a dummy forward pass to obtain the input number to the first fully connected layer
        dummydata = torch.rand(1, *self.input_img_size)
        dummydata = self.maxPool(F.relu(self.conv1(dummydata)))
        dummydata = self.maxPool(F.relu(self.conv2(dummydata)))
        dummydata = self.maxPool(F.relu(self.conv3(dummydata)))
        self.num_features_2fc = dummydata.shape[0] * dummydata.shape[1] * dummydata.shape[2] * dummydata.shape[
            3]  # there are clever ways but I don't recall them and need to search up
        self.fc1 = nn.Linear(self.num_features_2fc, 128)  # fully connected layers....
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # first convolution layer
        x = self.maxPool(x)
        x = F.relu(self.conv2(x))  # second convolution layer
        x = self.maxPool(x)
        x = F.relu(self.conv3(x))  # third convolution layer
        x = self.maxPool(x)
        # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# CNN model proposed by group 11
class NetG11(nn.Module):
    def __init__(self, input_img_size=(3, 256, 256)):
        super(NetG11, self).__init__()
        self.input_img_size = input_img_size
        self.conv1 = nn.Conv2d(self.input_img_size[0], 15, 5)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(15, 16, 5)
        # The input number to the first fully connected layer is a bit tricky
        # We can run through a dummy forward pass to obtain the input number to the first fully connected layer
        dummydata = torch.rand(1, *self.input_img_size)
        dummydata = self.maxPool(F.relu(self.conv1(dummydata)))
        dummydata = self.maxPool(F.relu(self.conv2(dummydata)))
        self.num_features_2fc = dummydata.shape[0] * dummydata.shape[1] * dummydata.shape[2] * dummydata.shape[
            3]  # there are clever ways but I don't recall them and need to search up
        self.fc1 = nn.Linear(self.num_features_2fc, 180)  # fully connected layers....
        self.fc2 = nn.Linear(180, 86)
        self.fc3 = nn.Linear(86, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # first convolution layer
        x = self.maxPool(x)
        x = F.relu(self.conv2(x))  # second convolution layer
        x = self.maxPool(x)
        # x = x.view(-1, 215296) # flatten the feature map into 1d, so that the fc layer can process it
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        if isinstance(criterion, nn.BCELoss):
            loss = criterion(output.view(-1), target.float())
        elif isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(output, target)
        else:
            raise Exception("Loss criterion is not defined! Please choose either BCELoss or CrossEntropyLoss.")
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / (batch_idx + 1)


def evaluate(model, device, eval_loader, criterion):
    model.eval()
    correct_eval, total_eval = 0, 0
    all_targets = torch.randn(0).to(device)
    all_predicted_labels = torch.randn(0).to(device)
    all_predicted_scores = torch.randn(0).to(device)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(eval_loader):
            print('Processing batch {0}...'.format(batch_idx))
            input, target = input.to(device), target.to(device)
            output = model(input)
            if isinstance(criterion, nn.BCELoss):
                predicted_label = (output.data > 0.5).float().view(-1)
                predicted_score = output.data.detach().view(-1)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted_label = torch.max(output.data.detach(), 1)
                predicted_score = F.softmax(output.data.detach(), dim=1)[:, 1]
            else:
                raise Exception("Loss criterion is not defined! Please choose either BCELoss or CrossEntropyLoss.")
            total_eval += predicted_label.size(0)
            correct_eval += (predicted_label == target).sum().item()
            all_targets = torch.cat((all_targets, target))
            all_predicted_labels = torch.cat((all_predicted_labels, predicted_label))
            all_predicted_scores = torch.cat((all_predicted_scores, predicted_score))

    all_targets = all_targets.cpu()
    all_predicted_labels = all_predicted_labels.cpu()
    all_predicted_scores = all_predicted_scores.cpu()

    import sklearn.metrics as metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ret = {}
    ret['accuracy'] = accuracy_score(all_targets, all_predicted_labels)
    ret['precision'] = precision_score(all_targets, all_predicted_labels, average='binary')
    ret['recall'] = recall_score(all_targets, all_predicted_labels, average='binary')
    ret['f1'] = f1_score(all_targets, all_predicted_labels, average='binary')
    fpr, tpr, threshold = metrics.roc_curve(all_targets, all_predicted_scores)
    ret['fpr'] = fpr
    ret['tpr'] = tpr
    ret['correct'] = correct_eval
    ret['total'] = total_eval
    ret['targets'] = all_targets
    ret['predictions'] = all_predicted_labels
    ret['predicted_scores'] = all_predicted_scores

    return ret


def predict_seed(seed_img, model, device, criterion):
    if not torch.is_tensor(seed_img) and isinstance(seed_img, np.ndarray):
        seed_img = cv2.resize(seed_img, (256, 256))
        # resize the image here (256, 256)
        # convert seed_img to tensor
        seed_img = transforms.ToTensor()(seed_img).float()
        if len(seed_img.shape) == 2:
            # convert (H,W) to (B,C,H,W)
            seed_img = seed_img.unsqueeze(0).unsqueeze(0)
        elif len(seed_img.shape) == 3:
            # convert (C,H,W) to (B,C,H,W)
            seed_img = seed_img.unsqueeze(0)
    else:
        raise Exception('The input image for classification must be either a Tensor or an Numpy array!')
    #seed_img = test_transform(seed_img)
    seed_img = seed_img.to(device)

    model.eval()
    with torch.no_grad():
        output = model(seed_img)
        if isinstance(criterion, nn.BCELoss):
            predicted_label = (output.data > 0.5).float().view(-1)
            predicted_score = output.data.detach().view(-1)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            _, predicted_label = torch.max(output.data.detach(), 1)
            predicted_score = F.softmax(output.data.detach(), dim=1)[:, 1]

    return predicted_label, predicted_score


class Arguments:
    k_folds = 5
    num_epochs = 40
    sched_step_size = 5
    gamma = 0.9
    lr = 1e-3
    batch_model_training = False  # only train the model as specified by which_net parameter when set False
                                  # If True, which_net, train_trans, and test_trans will be over-written by the main function
    which_net = 'G2-ClassToken'
    '''
    For my own models, choose from G1, G2, G2-dcd, G2-AU, G2-CBAM, G2-ClassToken, G2-MultiPath, G7, G11
    if choose from torchvision.models, there are following models:
    resnet, alexnet, squeezenet, vgg16, densenet, inception, googlenet, shufflenet, mobilenet, resnext, wideresnet, mnasnet
    '''
    if 'G2' in which_net:
        mask_type = 'conv' # choose from 'combined' or 'edge' or 'sprout' or 'sprout+edge' for G2-AU; 
                                # 'cbam' for G2-CBAM; 
                                # for 'G2-ClassToken', the mask_type is either 'conv' or 'outer' (Note: the performance of outer setting is poor)
    else:
        mask_type = which_net
    number_path = 2 # for multi-path model or ClassToken model only. for the latter number_path refers to the level of token (either 0, 1, or 2)
    number_conv_channels = 16 # for multi-path model or ClassToken only. for the latter number_conv_channels refers to the number of channels  of class token.
    use_colour_descriptor = False
    pretrained = True  # Only for pretrained models from torchvision.models
    feature_extract = True  # Only for pretrained models from torchvision.models
    train_trans = train_transforms   # use train_transforms for own models
    test_trans = test_transforms  # use test_transforms for own models
    #train_trans = train_transforms_for_torchvision_models   # use train_transforms_for_torchvision_models for pretrained models, use train_transforms_for_torchvision_inception for inception v3
    #test_trans = test_transforms_for_torchvision_models  # use test_transforms_for_torchvision_models for pretrained models, use test_transforms_for_torchvision_inception for inception v3
    train_trans_type = ''  # '', 'ToHSV', 'CatHSV', 'CatMask', 'CatMaskChannels'


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def create_net(which_net,
               pretrained=True, feature_extract=True,
               npath=4, conv_channel=16,
               mask_type='edge',
               use_colour_descriptor=True, num_channel=3):
    if which_net == 'G1':
        net = NetG1(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        criterion = nn.BCELoss()
    elif which_net == 'G2':
        net = NetG2(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), use_colour_descriptor=False)
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'G2-dcd':
        net = NetG2(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), use_colour_descriptor=True)
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'G2-AU':
        net = NetG2_AU(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mask_type=mask_type,
                       use_colour_descriptor=False)
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'G2-CBAM':
        net = NetG2_CBAM(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mask_type=mask_type,
                       use_colour_descriptor=False)
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'G2-ClassToken':
        net = NetG2_ClassToken(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                               token_level=npath,
                               token_channel=conv_channel,
                               mask_type=mask_type,
                               use_colour_descriptor=False)
        criterion = nn.CrossEntropyLoss()
        print('Class Tolen level:{0}, \t Number of class-token channels:{1}'.format(npath, conv_channel))
    elif which_net == 'G2-MultiPath':
        net = NetG2_MultiPath(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                              npath=npath, conv_channel=conv_channel,
                              mask_type=mask_type, use_colour_descriptor=False)
        criterion = nn.CrossEntropyLoss()
        print('Number of path:{0}, \t Number of channels in the first conv layer:{1}'.format(npath, conv_channel))
    elif which_net == 'G7':
        net = NetG7(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'G11':
        net = NetG11(input_img_size=(num_channel, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'resnet':
        net = models.resnet18(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last fc layer for binary classification
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in resnet18, so replacing fc is replacing the
        #     whole classifier. Consider layer4 and fc both as classifier.
        for param in net.layer4.parameters():
            param.requires_grad = True
        # (3) We can also replace the classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'alexnet':
        net = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_ftrs, out_features=2)
        # (2) Make the entire classifier trainable
        for param in net.classifier.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'squeezenet':
        net = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        net.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
        net.num_classes = 2
        # (2) There is only one conv layer in the classifier. Replacing the conv layer is replacing the whole classifer.
        #     Consider features.12 and classifier both as classifier
        for param in net.features[12].parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'vgg16':
        net = models.vgg16(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_ftrs, out_features=2)
        # (2) Make the entire classifier trainable
        #"""
        # Fine-tuning some high level features
        for i in range(26, 31):
            for param in net.features[i].parameters():
                param.requires_grad = True
        #"""
        for param in net.classifier.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'densenet':
        net = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in densenet, so replacing fc is replacing the
        #     whole classifier. Consider features.denseblock4 and classifer both as classifier
        for param in net.features.denseblock4.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'inception':
        """
        Inception v3, expects (299, 299) sized images and has auxiliary output
        """
        net = models.inception_v3(pretrained=pretrained, aux_logits=False)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        """
        # handle the auxiliary net
        num_ftrs = net.AuxLogits.fc.in_features
        net.AuxLogits.fc = nn.Linear(num_ftrs, out_features=2)
        """
        # handle the primary net
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in inception, so replacing fc is replacing the
        #     whole classifier. Consider Mixed_7a, Mixed_7b, Mixed_7c and fc as classifier
        for param in net.Mixed_7a.parameters():
            param.requires_grad = True
        for param in net.Mixed_7b.parameters():
            param.requires_grad = True
        for param in net.Mixed_7c.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'googlenet':
        """
        GoogLeNet, has 2 auxiliary outputs
        """
        net = models.googlenet(pretrained=pretrained, init_weights=True)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        """
        # handle the first auxiliary net
        num_ftrs = net.aux1.fc2.in_features
        net.aux1.fc2 = nn.Linear(num_ftrs, out_features=2)
        # handle the second auxiliary net
        num_ftrs = net.aux2.fc2.in_features
        net.aux2.fc2 = nn.Linear(num_ftrs, out_features=2)
        """
        # handle the primary net
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in googlenet, so replacing fc is replacing the
        #     whole classifier. Consider inception5a, inception5b, and fc as classifier
        for param in net.inception5a.parameters():
            param.requires_grad = True
        for param in net.inception5b.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'shufflenet':
        net = models.shufflenet_v2_x1_0(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in shufflenet, so replacing fc is replacing the
        #     whole classifier. Consider conv5 and fc as classifier
        for param in net.conv5.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'mobilenet':
        net = models.mobilenet_v2(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in densenet, so replacing fc is replacing the
        #     whole classifier. Consider features[18] and classifier layer as classifier
        for param in net.features[18].parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'resnext':
        net = models.resnext50_32x4d(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in densenet, so replacing fc is replacing the
        #     whole classifier. Consider layer4 and fc both as classifier.
        for param in net.layer4.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'wideresnet':
        net = models.wide_resnet50_2(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in densenet, so replacing fc is replacing the
        #     whole classifier. Consider layer4 and fc both as classifier.
        for param in net.layer4.parameters():
            param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    elif which_net == 'mnasnet':
        net = models.mnasnet1_0(pretrained=pretrained)
        set_parameter_requires_grad(net, feature_extract)
        # The following (1) is a must, but (2) and (3) are optional. (2) and (3) are alternatives and should not appear
        # at the same time in the network architecture
        # (1) Modify the last layer of the classifier for binary classification
        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, out_features=2)
        # (2) Note that there is only one Linear layer as the classifier in densenet, so replacing fc is replacing the
        #     whole classifier. Consider layer[13] to layer[16] and classifier layer both as classifier.
        for i in range(10, 17):
            for param in net.layers[i].parameters():
                param.requires_grad = True
        # (3) Replace the entire classifier with another classifier architecture
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("The Neural Network Model has to be one of the following: G1, G2, G7, G11, or models from torchvision.models")

    return net, criterion


def load_trained_model(model=NETWORK_MODEL):
    if model:  # make prediction for each seed if a model is specified
        # obtain model parameters
        model_type = model[0]
        trained_model = model[1]
        #mask_type = model[2]
        mask_type = 'edge'
        # create network model
        if model_type == 'G2-MultiPath' or model_type == 'G2-ClassToken':
            # example model name would be 'G2-MultiPath_4_16_07-25 07:23:50_out.pth' or 'G2-MultiPath__07-25 07:17:24_out.pth'
            name_list = trained_model.rsplit('/', 1)[1].split('_', 3)
            # The above would result in ['G2-MultiPath', '4', '16', '07-25 07:23:50_out.pth'] for the former
            # and ['G2-MultiPath', '', '07-25 07:17:24_out.pth'] for the latter
            # We would like to extract 4 and 16 for the former and use default 4 and 16 for the latter
            if name_list[1] == '':
                npath = 4
                conv_channel = 16
            else:
                npath = int(name_list[1])
                conv_channel = int(name_list[2])

            net, criterion = create_net(model_type,
                                        npath=npath,
                                        conv_channel=conv_channel,
                                        mask_type=mask_type,
                                        use_colour_descriptor=USE_COLOUR_DESCRIPTOR)
        else:
            net, criterion = create_net(model_type,
                                        pretrained=True,
                                        mask_type=mask_type,
                                        use_colour_descriptor=USE_COLOUR_DESCRIPTOR)
        # load weights from a trained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(trained_model))
        else:
            net.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
        return net, device, criterion
    else:
        raise Exception('Network model is not specified!')


def main(args):
    #args = Arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #dataset_train = datasets.ImageFolder(root=os.path.join(WORKING_DIR, "dataset/train"), transform=train_transforms)
    #dataset_validation = datasets.ImageFolder(root=os.path.join(WORKING_DIR, "dataset/validation"),
    #                                          transform=transform_test)
    #dataset_test = datasets.ImageFolder(root=os.path.join(WORKING_DIR, "dataset/test"), transform=transform_test)
    train_df = pd.read_csv('C:/Users/User/Desktop/AAR/csv/batch1_train_cropped_seeds.csv')
    validation_df = pd.read_csv('C:/Users/User/Desktop/AAR/csv/batch1_validation_cropped_seeds.csv')
    test_df = pd.read_csv('C:/Users/User/Desktop/AAR/csv/batch1_test_cropped_seeds.csv')
    #train_df = pd.read_csv('/content/drive/My Drive/AAR/csv/batch1_train_cropped_seeds.csv')
    #validation_df = pd.read_csv('/content/drive/My Drive/AAR/csv/batch1_validation_cropped_seeds.csv')
    #test_df = pd.read_csv('/content/drive/My Drive/AAR/csv/batch1_test_cropped_seeds.csv')
    
    dataset_train = IndividualSeedDataset(train_df, transform=args.train_trans)
    dataset_validation = IndividualSeedDataset(validation_df, transform=args.test_trans)
    dataset_test = IndividualSeedDataset(test_df, transform=args.test_trans)

    best_model = None
    best_total_val = 0
    best_correct_val = 0
    best_model_acc = 0

    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        sampler=RandomSampler(dataset_train),
        num_workers=4
    )

    val_loader = DataLoader(
        dataset_validation,
        batch_size=64,
        num_workers=4
    )

    if args.train_trans_type == 'ToHSV' or args.train_trans_type == '':
        num_channels = 3
    elif args.train_trans_type == 'CatHSV' or args.train_trans_type == 'CatMaskChannels':
        num_channels = 6
    elif args.train_trans_type == 'CatMask':
        num_channels = 4
    else:
        raise Exception('Transform type not defined!')

    print(args.mask_type)
    net, criterion = create_net(args.which_net,
                                npath=args.number_path,
                                conv_channel=args.number_conv_channels,
                                mask_type=args.mask_type,
                                use_colour_descriptor=args.use_colour_descriptor,
                                num_channel=num_channels)
    net = net.to(device)
    # Write logs
    time_of_run = time.strftime('%m-%d %H:%M:%S', time.localtime())
    if args.which_net == 'G2' and args.use_colour_descriptor:
        experiment = args.which_net + '_dcd' + '_' + args.train_trans_type
    elif args.which_net == 'G2-AU' and args.use_colour_descriptor:
        experiment = args.which_net + '_dcd_' + args.mask_type + '_' + args.train_trans_type
    elif args.which_net == 'G2-AU':
        experiment = args.which_net + '-' + args.mask_type + '_' + args.train_trans_type
    elif args.which_net == 'G2-MultiPath':
        experiment = args.which_net + '_' + str(args.number_path) + '_' + str(args.number_conv_channels)
    elif args.which_net == 'G2-ClassToken':
        experiment = args.which_net + args.mask_type.upper() + '_' + str(args.number_path) + '_' + str(args.number_conv_channels)
    elif args.which_net in {'resnet', 'alexnet', 'squeezenet', 'vgg16', 'densenet', 'inception', 'googlenet',
                            'shufflenet', 'mobilenet', 'resnext', 'wideresnet', 'mnasnet'}:
        experiment = args.which_net + '_pretrained_' + str(args.pretrained) + '_feature_extracting_' + str(args.feature_extract)
    else:
        experiment = args.which_net + '_' + args.train_trans_type
    experiment = experiment + '_' + time_of_run
    #logname = os.path.join('runs', experiment)
    #logname = logname.replace(":", "_")
    logname = 'runs/' + experiment
    writer = SummaryWriter(logname)

    # http://karpathy.github.io/2019/04/25/recipe/
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # criterion = nn.BCELoss()
    scheduler = StepLR(optimizer, step_size=args.sched_step_size, gamma=args.gamma)

    num_epoch = 0
    num_epoch_not_improved = 0
    while num_epoch_not_improved < 20:
        num_epoch += 1

        # train
        print(f'Epoch number {num_epoch}')
        train_loss = train(net, device, train_loader, optimizer, criterion)
        print(f'Loss: {train_loss}')
        writer.add_scalar('Train Loss', train_loss, num_epoch)
        scheduler.step()

        # evaluate
        output_val = evaluate(net, device, val_loader, criterion)
        acc = output_val['accuracy']
        correct_val = output_val['correct']
        total_val = output_val['total']
        print('\t{0:.2f}% ({1}/{2})'.format(acc * 100, correct_val, total_val))
        writer.add_scalar('Validation Accuracy', acc, num_epoch)
        writer.add_scalar('Validation Precision', output_val['precision'], num_epoch)
        writer.add_scalar('Validation Recall', output_val['recall'], num_epoch)
        writer.add_scalar('Validation f1 score', output_val['f1'], num_epoch)

        if acc > best_model_acc:
            best_model = copy.deepcopy(net)
            best_model_acc = acc
            best_total_val = total_val
            best_correct_val = correct_val
            num_epoch_not_improved = 0
        else:
            num_epoch_not_improved += 1

        print('Current best model validation:')
        print('\t{0:.2f}% ({1}/{2})'.format(best_model_acc * 100, best_correct_val, best_total_val))
        print()
        writer.add_scalar('Best Model Accuracy', best_model_acc, num_epoch)

    test_loader = DataLoader(
        dataset_test,
        batch_size=64,
        num_workers=4
    )

    torch.save(best_model.state_dict(), os.path.join(WORKING_DIR, 'models', experiment + '_' + 'out.pth'))

    print('Evaluate on training set')
    output_train = evaluate(best_model, device, train_loader, criterion)
    roc_auc_train = metrics.auc(output_train['fpr'], output_train['tpr'])
    print('Accuracy: \t{0:.2f}% ({1}/{2})'.format(output_train['accuracy'] * 100, output_train['correct'],
                                                  output_train['total']))
    print('Precision: \t{0:.2f}, Recall: \t{1:.2f}, F1: \t{2:.2f}, AUC: \t{3:.2f}'.format(output_train['precision'],
                                                                                          output_train['recall'],
                                                                                          output_train['f1'],
                                                                                          roc_auc_train))

    print('Evaluate on test set')
    output_test = evaluate(best_model, device, test_loader, criterion)
    roc_auc_test = metrics.auc(output_test['fpr'], output_test['tpr'])
    print('Accuracy: \t{0:.2f}% ({1}/{2})'.format(output_test['accuracy'] * 100, output_test['correct'],
                                                  output_test['total']))
    print('Precision: \t{0:.2f}, Recall: \t{1:.2f}, F1: \t{2:.2f}, AUC: \t{3:.2f}'.format(output_test['precision'],
                                                                                          output_test['recall'],
                                                                                          output_test['f1'],
                                                                                          roc_auc_test))

    '''### for debugging only ####
    for i in range(len(output_test['targets'])):
      print('The {0:d}-th testing sample:\n \
      True label:{1:.2f},\tPredicted Label:{2:.2f},\tPredicted score:{3:.2f}'.format(
          i+1,
          output_test['targets'][i],
          output_test['predictions'][i],
          output_test['predicted_scores'][i]))
    ###########################'''

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(output_test['fpr'], output_test['tpr'], 'b', label='AUC = %0.2f' % roc_auc_test)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    writer.add_pr_curve('Test pr_curve', output_test['targets'], output_test['predicted_scores'], 0)
    writer.add_scalar('Test accuracy', output_test['accuracy'], 0)
    writer.add_scalar('Test precision', output_test['precision'], 0)
    writer.add_scalar('Test recall', output_test['recall'], 0)
    writer.add_scalar('Test f1 score', output_test['f1'], 0)
    writer.add_scalar('Test AUC', roc_auc_test, 0)
    writer.close()


# Define list of models that will be trained in batch mode
TORCHVISION_MODEL_NAMES = ['resnet', 'alexnet', 'squeezenet', 'vgg16', 'densenet', 'shufflenet', 'mobilenet', 'resnext',
                           'wideresnet', 'mnasnet', 'googlenet', 'inception']


if __name__ == '__main__':
    args = Arguments()
    if args.batch_model_training:
        # go through all the models in MODEL_NAMES to set args.which_net, args.train_trans, and args.test_trans
        for model_name in TORCHVISION_MODEL_NAMES:
            args.which_net = model_name
            if model_name == 'inception':
                args.train_trans = train_transforms_for_torchvision_inception
                args.test_trans = test_transforms_for_torchvision_inception
            else:
                args.train_trans = train_transforms_for_torchvision_models
                args.test_trans = test_transforms_for_torchvision_models
            main(args=args)
    else:
        main(args=args)

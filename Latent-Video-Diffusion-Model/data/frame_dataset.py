import os
import random
import re # for regular expressions
from PIL import ImageFile
from PIL import Image
import torch 
import torch.utils.data as data 
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video # transformation designed for video data

""" VideFrame Dataset"""
ImageFile.LOAD_TRUNCATED_IMAGES = True # it instructs pillow to load images even if they are truncated (incomplete or truncated)
# This can be useful when deadling with a large dataset where some images might be partially downloaded or corrupted
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 
] # file extensions of image fles that will be considered valid for the dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
    '''
    Im = Image.open(path)
    return Im.convert('RGB')

def accimage_loader(path):
    import accimage   
    try:
        return accimage.Image(path)
    
    except IOError: 
        # potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
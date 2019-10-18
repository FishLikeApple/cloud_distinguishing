import os 
import argparse 
import torch 
from datasets import CloudDataset 
from torch.utils.data import DataLoader 
import torch.nn as nn 
from optimizers import RAdam 
from utils import Metric
from tqdm import tqdm 
import torch.optim as optim
from models.model import Model
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from models.loss import Criterion
from datasets.dataset import null_collate
import numpy as np
import pandas as pd
import cv2
from albumentations import (
    Compose,
    Resize
)

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--test_dataset', default='./data/test_images', type=str, help='config file path')
parser.add_argument('--checkpoint', default='./checkpoint.pth', type=str, help='config file path')
parser.add_argument('--list_test', default='./data/test.csv', type=str)
parser.add_argument('--submission', default='./submission.csv', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_class', default=5, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--encoder', default="resnet34", type=str)
parser.add_argument('--decoder', default="hrnet", type=str)  
parser.add_argument('--mode', default='non-cls', type=str)
args = parser.parse_args()

# some hyperparms
original_height = 1400
original_width = 2100
objective_height = 350
objective_width = 525
type_list = ['Fish', 'Flower', 'Gravel', 'Sugar']

test_dataset = CloudDataset(root_dataset = args.test_dataset, list_data = args.list_test, phase='test', mode=args.mode)

model = Model(num_class=args.num_class, encoder = args.encoder, decoder = args.decoder, mode=args.mode)
model = model.cuda()
model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
model.eval()
criterion = Criterion(mode=args.mode)

test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)

submission = pd.read_csv(args.list_test)

def get_transforms():
    list_transforms = []
    list_transforms.extend(
        [
            Resize(height=objective_height, width=objective_width,  interpolation=cv2.INTER_NEAREST)
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

# the function is from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def output2rle(mask_output, type=1):
    '''change a certain type of model mask output to the submission type'''
    
    mask_output = np.asarray(mask_output)
    mask = np.where(mask_output==type, 1, 0)
    return mask2rle(mask)

def test(data_loader):
    model.eval()
    transform_fn = get_transforms()
    for img, segm, img_id in tqdm(data_loader):
        img = img.cuda()
        output = np.argmax(model(img).cpu().detach().numpy(), axis=1)
        mask = transform_fn(image=np.squeeze(output))['image']
        for i, type in enumerate(type_list):
            rle = output2rle(mask, i)
            submission.loc[submission['Image_Label']==img_id[0]+'_'+type, 'EncodedPixels'] = rle
    submission.to_csv(args.submission, index=False)

test(test_loader)

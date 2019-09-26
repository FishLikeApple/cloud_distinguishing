import os 
os.system('pip install -r Steel-Defect-Detection/requirements.txt')
import argparse 
import torch 
from datasets import SteelDataset 
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

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--test_dataset', default='./data/test_images', type=str, help='config file path')
parser.add_argument('--checkpoint', default='./checkpoint.pth', type=str, help='config file path')
parser.add_argument('--list_test', default='./data/test.csv', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_class', default=5, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--encoder', default="resnet34", type=str)
parser.add_argument('--decoder', default="hrnet", type=str)  
parser.add_argument('--mode', default='non-cls', type=str)
args = parser.parse_args()

args = parser.parse_args()

test_dataset = SteelDataset(root_dataset = args.test_dataset, list_data = args.list_test, phase='test', mode=args.mode)

model = Model(num_class=args.num_class, encoder = args.encoder, decoder = args.decoder, mode=args.mode)
model = model.cuda()
model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
model.eval()
criterion = Criterion(mode=args.mode)

test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)

# the above function is from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
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
    
# the above function is from https://www.kaggle.com/bibek777/heng-s-model-inference-kernel
def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def test(data_loader):
    model.eval()
    total_loss = 0
    accumulation_steps = 32 // args.batch_size
    for idx, (img, segm) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        outputs = model(img)
    print('outputs')  
    print(outputs[0])
        
    torch.cuda.empty_cache()
    return total_loss/len(data_loader)

test(test_loader)

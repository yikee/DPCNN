import argparse
import shutil
from torch.utils.data import DataLoader
from utils import *
from train import *
from config import Config
import json
import numpy

parser = argparse.ArgumentParser(description='DPCNN Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='DPCNN', choices='DPCNN', help='model architecture: (default: DPCNN)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

def main():
    args = parser.parse_args()
    opt = Config()
    with open(opt.TRAIN_DATASET_PATH, 'r') as f:
        training_set = [json.loads(line) for line in f]
        training_set = [[label+1, numpy.array(image, dtype=numpy.float32)] for label, image in training_set]
    with open(opt.TEST_DATASET_PATH, 'r') as f:
        test_set = [json.loads(line) for line in f]
        test_set = [[label+1, numpy.array(image, dtype=numpy.float32)] for label, image in test_set]
    opt.NUM_CLASSES = 3

    #print(len(training_set))
    #print(len(test_set))
    
    train_loader = DataLoader(dataaset=training_set, batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)

    net = training(train_loader, test_loader, opt)

if __name__ == '__main__':
    main()
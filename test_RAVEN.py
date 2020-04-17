import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = 1280,1280
import numpy as np
import math
import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from model_RAVEN import Model
from torch.utils.data import DataLoader
import torch.nn.utils
import matplotlib.pyplot as plt
import itertools
from data_utility import ToTensor
from data_utility import dataset_raven as dataset


def test(model,testloader,validloader,device,args):
    model.eval()
    for epoch in range(1):
        if args.valid_result:
            total_correct = 0.0
            for (data,label,meta_target) in validloader:
                data = data.view(-1,16,args.image_size,args.image_size)
                label = label.view(-1)
                meta_target = meta_target.view(-1,9)
                data = data.to(device)
                label = label.to(device)
                meta_target = meta_target.to(device)
                _,score_vec = model(data,label,meta_target)
                _,pred = torch.max(score_vec,1)
                c = (pred == label).squeeze()  
                total_correct += torch.sum(c).item()
            accuracy = total_correct/(validData.__len__())
            print('Validation Accuracy:',accuracy)
      
        total_correct = 0.0
        for (data,label,meta_target) in testloader:
            data = data.view(-1,16,args.image_size,args.image_size)
            label = label.view(-1)
            meta_target = meta_target.view(-1,9)
            data = data.to(device)
            label = label.to(device)
            meta_target = meta_target.to(device)
            _,score_vec = model(data,label,meta_target)
            _,pred = torch.max(score_vec,1)
            c = (pred == label).squeeze()                
            meta_target_np = meta_target.cpu().numpy()   
            batch_correct = torch.sum(c).item()
            total_correct += batch_correct
            
        accuracy = total_correct/(testloader.dataset.__len__())
        print('Test Accuracy:',accuracy)


def main():
    parser = argparse.ArgumentParser(description='RAVEN test args')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--image-size', type=float, default=80, metavar='IMSIZE',
                        help='input image size (default: 80)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='parallel training on multiple GPUs')
    parser.add_argument('--valid_result', action='store_true', default=False,
                        help='compute results on validation dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-path-name', default='', type=str, metavar='PATH',
                        help='The path+name of model to be loaded')
    args = parser.parse_args()


    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cpu" if args.no_cuda else "cuda")

    test_data = dataset(args.data, "test", args.image_size, transform=transforms.Compose([ToTensor()]))
    valid_data = dataset(args.data, "val", args.image_size, transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8)
    validloader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=8)

    model = Model(args.image_size,args.image_size)



    if not args.no_cuda:
        model.cuda()


    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path_name))


    test(model,testloader,validloader,device,args)

if __name__ == '__main__':
    main()

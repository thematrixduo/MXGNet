import os
import argparse
import itertools

import numpy as np
import math
import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from model_PGM import Model
from torch.utils.data import DataLoader
import torch.nn.utils

from data_utility import ToTensor
#For preprocessed data loading,
from data_utility import dataset
from radam import RAdam


model_save_name = 'PGM_ori_best.pt'
optimizer_save_name = 'PGM_ori_best_opt.pt'

def train(model,optimizer,trainloader,validloader,device,args):
    avg_loss = 0
    count = 0
    best_acc = 0
    for epoch in range(args.epochs):
        for (data,label,meta_target) in trainloader:

            data = data.view(-1,16,args.image_size,args.image_size)
            label = label.view(-1)
            meta_target = meta_target.view(-1,12)
            bs = data.size()[0]
            data = data.to(device)
            label = label.to(device)
            meta_target = meta_target.to(device)
            optimizer.zero_grad()
            loss,_ = model(data,label,meta_target)
            loss = torch.sum(loss)
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step()
            count += 1
            if count % 100 == 0:
                print('Epoch-{}; Count-{}; loss: {} '.format(epoch, count, avg_loss / 100))
                avg_loss = 0

        if epoch > 0:
            model.eval()
            total_correct = 0.0
            num_samples = 0
            for (data,label,meta_target) in validloader:
                data = data.view(-1,16,args.image_size,args.image_size)
                label = label.view(-1)
                num_samples+=label.size(0)
                meta_target = meta_target.view(-1,12)
                data = data.to(device)
                label = label.to(device)
                meta_target = meta_target.to(device)
                _,score_vec = model(data,label,meta_target)
                _,pred = torch.max(score_vec,1)
                c = (pred == label).squeeze()   
                total_correct += torch.sum(c).item()
            accuracy = total_correct/num_samples
            print('Accuracy:',accuracy,total_correct,num_samples)   
            model.train()

        if epoch > 0 and accuracy > best_acc and args.save_model:
            print('saving model')
            torch.save(model.state_dict(), os.path.join(args.model_save_path,model_save_name ))
            torch.save(optimizer.state_dict(), os.path.join(args.model_save_path,optimizer_save_name ))
            best_acc = accuracy

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--batch-size-val', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--image-size', type=float, default=80, metavar='IMSIZE',
                        help='input image size (default: 80)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='parallel training on multiple GPUs')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-save-path', default='', type=str, metavar='PATH',
                        help='For Saving the current Model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()


    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cpu" if args.no_cuda else "cuda")

    train_data = dataset(args.data, "train", args.image_size, transform=transforms.Compose([ToTensor()]),shuffle=True)
    valid_data = dataset(args.data, "val", args.image_size, transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    validloader = DataLoader(valid_data, batch_size=args.batch_size_val, shuffle=False, num_workers=8)

    model = Model(args.image_size,args.image_size)

    optimizer = RAdam(model.parameters(),lr=args.lr,weight_decay = 1e-8)

    if not args.no_cuda:
        model.cuda()


    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.resume,model_save_name )))
        optimizer.load_state_dict(torch.load(os.path.join(args.resume,optimizer_save_name )))

    train(model,optimizer,trainloader,validloader,device,args)

if __name__ == '__main__':
    main()

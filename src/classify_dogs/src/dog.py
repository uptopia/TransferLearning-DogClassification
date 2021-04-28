# -*- coding: utf-8 -*- #
# ===============================
#   PyTorch Transfer Learning
#        PyTorch 轉移學習
# ===============================
# written by Shang-Wen, Wong. (2021.4.21)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

# from torchviz import make_dot

import numpy as np

import matplotlib.pyplot as plt

import time
import os
import copy

num_class = 4

# PATH = 'ismodel.pth'
# PATH = 'model_dog1.pth'
PATH = '/home/upup/transfer_learning_ws/weights/model_dog1.pth'
IMG_FOLDER_PATH = "/home/upup/transfer_learning_ws/dog_data/"


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

def imshow(inp, title = None):
    
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
    if title is not None:
        plt.title(title)

    plt.pause(1)#1)#.01)

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {0}/{1}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        #每個epoch中訓練和驗證的部份
        for phase in ['train', 'val']:
            if phase == 'train': 
                scheduler.step()
                model.train(True)
            else: 
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                #使用GPU與否
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                #初始化梯度值
                optimizer.zero_grad()

                #前向
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #後向
                if phase == 'train': 
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data #[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images = 6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        print('preds', preds)

        for j in range(inputs.size()[0]):
            print('j', j)
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_name[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


def predict_image(model, data):

    inputs, labels = data
    # print("labels",labels)
    print('inputs', np.size(inputs))

    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    # print("outputs", outputs.data)
    _, preds = torch.max(outputs.data, 1)
    
    for j in range(inputs.size()[0]):
        print('predicted: {}, {}'.format(class_names[preds[j]], j))
        imshow(inputs.cpu().data[j])


if __name__ == '__main__':
    # #===========#
    # # 數據前處理
    # #===========#
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ]),

    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ]),
    # }

    # data_dir = 'dog_data'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
    #                 for x in ['train', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle = True, num_workers = 4) 
    #                 for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # class_name = image_datasets['train'].classes
    use_gpu = torch.cuda.is_available()
    print("gpu available = {0}".format(use_gpu))

    # inputs, classes = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs)

    # # imshow(out, title=[class_name[x] for x in classes])

    # # #============#
    # # # FINETUNING
    # # #============#
    # # model_fit = models.resnet18(pretrained = True)
    # # num_ftrs = model_fit.fc.in_features
    # # model_fit.fc = nn.Linear(num_ftrs, num_class)
    # # # visualize_model(model_fit)

    # # if use_gpu:
    # #     model_fit = model_fit.cuda()

    # # criterion = nn.CrossEntropyLoss()

    # # #模型中[所有參數]都可以被優化
    # # optimizer_fit = optim.SGD(model_fit.parameters(), lr = 0.001, momentum = 0.9)

    # # #學習率(LR) 每step_size個epoch下降gamma
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_fit, step_size = 7, gamma = 0.1)

    # # #對已經訓練好的模型參數繼續訓練，在每個epoch中紀錄此刻最好的模型參數
    # # model_fit = train_model(model_fit, criterion, optimizer_fit, exp_lr_scheduler, num_epochs = 5)

    # # #=====================#
    # # # Save and Load Model
    # # #=====================#
    # # torch.save(model_fit, PATH)

    # # model_input = torch.load(PATH)
    # # model_input.eval()
    # # # for param in model_input.parameters():
    # # #     print(param)
    # # visualize_model(model_input)


    # #==========================#
    # # FIXED FEATURE EXTRACTOR
    # #==========================#
    # model_conv = torchvision.models.resnet18(pretrained = True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False #False: 梯度不會進行更新
    
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, num_class)

    # if use_gpu:
    #     model_conv = model_conv.cuda()
    
    # criterion = nn.CrossEntropyLoss()

    # #模型中[只有FC]可以被優化 
    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr = 0.001, momentum = 0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size = 7, gamma = 0.1)
    # model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs = 5)

    # #=====================#
    # # Save and Load Model
    # #=====================#
    # torch.save(model_conv, PATH)

    model_input = torch.load(PATH, map_location=torch.device('cpu'))    
    model_input.eval()
    # for param in model_input.parameters():
    #     print(param)
    # visualize_model(model_input)


    data_transforms_pred = {
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # data_dir = 'dog_data'
    image_datasets_pred = {x: datasets.ImageFolder(os.path.join(IMG_FOLDER_PATH, x), data_transforms_pred[x]) 
                    for x in ['val']}
    dataloaders_pred = {x: torch.utils.data.DataLoader(image_datasets_pred[x], batch_size = 1, shuffle = False, num_workers = 4) 
                    for x in ['val']}

    class_names = image_datasets_pred['val'].classes
    for n, data_pred in enumerate(dataloaders_pred['val']):
        print("n", n)
        print("data_pred", data_pred)
        predict_image(model_input, data_pred)

# # Reference:
# # 轉移學習 https://blog.csdn.net/IAMoldpan/article/details/78635981
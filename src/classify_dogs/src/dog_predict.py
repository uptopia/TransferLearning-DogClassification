#! /usr/bin/env python3

# -*- coding: utf-8 -*- #
# ==========================================
#   PyTorch Transfer Learning(Prediction)
#        PyTorch 轉移學習(預測)
# ==========================================
# written by Shang-Wen, Wong. (2021.4.23)

import rospy
from std_msgs.msg import String
# from sensor_msgs.msg import Image

import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import os
import numpy as np

from sensor_msgs.msg import Image as SensorImage

from PIL import Image as PILImage


import cv2

import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/') #cv_bridge with python3 (self-build)
from cv_bridge import CvBridge, CvBridgeError

#============#
# PARAMETERS
#============#
IMG_FOLDER_PATH = "/home/upup/transfer_learning_ws/dog_data/"
MODEL_SAVE_PATH = '/home/upup/transfer_learning_ws/weights/model_dog1.pth'


# #要更改的參數
#resnet18
# 'train', 'val'
# num_class
# lr
# optimizer
# batch_size
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])


# # def predict():

# #     return 

# def predict_image(model, data):

#     inputs, labels = data
#     print("labels", labels)

#     if use_gpu:
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#     else:
#         inputs, labels = Variable(inputs), Variable(labels)

#     outputs = model(inputs)
#     # print("outputs", outputs.data)
#     _, preds = torch.max(outputs.data, 1)
    
#     for j in range(inputs.size()[0]):
#         print('predicted: {}, {}'.format(class_names[preds[j]], j))
#         imshow(inputs.cpu().data[j])


# def classify_dogs(data):
#     bridge = CvBridge()
#     try:
#         cv_image = bridge.imgmsg_to_cv2(data, data.encoding)
#     except CvBridgeError as e:
#         print(e)

#     pub_dog_class.()

def ppp():
    tensor=torch.from_numpy(np.array(I)).permute(2,0,1).float()/255.0 #np.asarray()
    tensor=tensor.reshape((1,3,224,224))
    tensor=tensor.to(device)
    #print(tensor.shape)
    output = model_input(tensor)
    print(output)
    _, pred = torch.max(output.data,1)
    # print('predicted: {}'.format(class_names[pred]))
    print('predicted: {}'.format(pred))

    return

def image_cb(data):
    print("image_cb")
    # print(data)
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, data.encoding)
        cv2.imshow('Color', cv_image)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print(e)
        return

if __name__ == "__main__":
    print('test')
    print(torch.__version__)
    rospy.init_node("classify_dog_node")

    sub_img = rospy.Subscriber('publish_img', SensorImage, image_cb)

    rospy.spin()

    # #=========#
    # # 載入模型
    # #=========#
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_input = torch.load(MODEL_SAVE_PATH, map_location = torch.device(device))
    # model_input.eval()


    # class_names = datasets.ImageFolder(os.path.join(IMG_FOLDER_PATH+'val')).classes
    # print(class_names)
    
    # # # IMG_PATH = "/home/upup/transfer_learning_ws/dog_data/val/n02085620-Chihuahua/n02085620_4159.jpg"
    # # # IMG_PATH = "/home/upup/transfer_learning_ws/dog_data/val/n02111889-Samoyed/n02111889_1444.jpg"
    # # # IMG_PATH = "/home/upup/transfer_learning_ws/dog_data/val/n02105855-Shetland_sheepdog/n02105855_4048.jpg"
    # IMG_PATH = "/home/upup/transfer_learning_ws/dog_data/val/n02116738-African_hunting_dog/n02116738_2020.jpg"
 
    # I = Image.open(IMG_PATH) 
    # I.show()    
    # # I.save('./save.png')
    # I_array = np.array(I) #np.array(); np.asarray()
    # # print(I.shape)
    # print(I_array.shape)

    # # print("当前模型准确率为：",model_input["epoch_acc"])
    # ppp()
   
# #        

# #     image_datasets_pred = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms_pred[x]) 
# #                     for x in ['val']}
# #     dataloaders_pred = {x: torch.utils.data.DataLoader(image_datasets_pred[x], batch_size = 1, shuffle = False, num_workers = 4) 
# #                     for x in ['val']}

# #     class_names = image_datasets_pred['val'].classes
# #     for n, data_pred in enumerate(dataloaders_pred['val']):
# #         print("n", n)
# #         print("data_pred", data_pred)
# #         predict_image(model_input, data_pred)



# #     # sub_rgb = rospy.Subscriber('', Image, classify_dogs)
# #     # pub_dog_class = rospy.Publisher('dog_class', String, queue_size = 1)

# #     # rospy.spin()


# # Reference:
# # PyTorch讀圖 https://cloud.tencent.com/developer/article/1685951
# https://blog.csdn.net/hjxu2016/article/details/79104607
# cv_bridge把sensor_msgs/Image: https://blog.csdn.net/weixin_43434136/article/details/112646275
# ROS webcam: https://www.ncnynl.com/archives/201703/1437.html
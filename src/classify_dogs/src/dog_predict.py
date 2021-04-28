#! /usr/bin/env python3

# -*- coding: utf-8 -*- #
# ==========================================
#   PyTorch Transfer Learning(Prediction)
#        PyTorch 轉移學習(預測)
# ==========================================
# written by Shang-Wen, Wong. (2021.4.23)

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as SensorImage

import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import os
import numpy as np

from PIL import Image as PILImage

import cv2
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/') #cv_bridge with python3 (self-build)
from cv_bridge import CvBridge, CvBridgeError

#============#
# PARAMETERS
#============#
IMG_FOLDER_PATH = "/home/upup/TransferLearning-DogClassification/dog_data/"
MODEL_SAVE_PATH = '/home/upup/TransferLearning-DogClassification/weights/model_dog1.pth'

# #要更改的參數
#resnet18
# 'train', 'val'
# num_class
# lr
# optimizer
# batch_size
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

#=========#
# 載入模型
#=========#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_input = torch.load(MODEL_SAVE_PATH, map_location = torch.device(device))
model_input.eval()

class_names = datasets.ImageFolder(os.path.join(IMG_FOLDER_PATH+'val')).classes
# print(class_names)

preprocess_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def classify_dogs(array_img):
    inputs = preprocess_transform(array_img)
    inputs_unsqueeze = inputs.unsqueeze(0)
    inputs_unsqueeze = inputs_unsqueeze.to(device)
    outputs = model_input(inputs_unsqueeze)
    _, preds = torch.max(outputs.data, 1) #_, indices = torch.sort(outputs, descending = True)

    for j in range(inputs_unsqueeze.size()[0]):
        text = 'predicted: ' + class_names[preds[j]]
        print(text)

    return text

def image_cb(data):
    print("image_cb")

    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")#data.encoding)
    except CvBridgeError as e:
        print(e)

    array_img = PILImage.fromarray(cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB))

    text = classify_dogs(array_img)

    #publish dog class
    pub_dog_class.publish(text)

    #show dog image
    cv2.putText(cv_image, text, (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Dog',cv_image)
    cv2.waitKey(1)

    return

if __name__ == "__main__":
    print('test')
    # print(torch.__version__)
    # print(cv2.__version__)

    rospy.init_node("classify_dog_node")

    sub_img = rospy.Subscriber('publish_img', SensorImage, image_cb)
    pub_dog_class = rospy.Publisher('dog_class', String, queue_size = 1)

    rospy.spin()

# # Reference:
# # PyTorch讀圖 https://cloud.tencent.com/developer/article/1685951
# https://blog.csdn.net/hjxu2016/article/details/79104607
# cv_bridge把sensor_msgs/Image: https://blog.csdn.net/weixin_43434136/article/details/112646275
# ROS webcam: https://www.ncnynl.com/archives/201703/1437.html
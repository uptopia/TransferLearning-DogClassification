#! /usr/bin/env python3

# -*- coding: utf-8 -*- #
# ==========================================
#   PyTorch Transfer Learning(Prediction)
#        PyTorch 轉移學習(預測)
# ==========================================
# written by Shang-Wen, Wong. (2021.4.23)

# [Methods]
# 1. Read image from computer
# 2. Stream Webcam
# 3. Stream Realsense D435i

#opencv 讀圖-> sensor_msgs -> PIL


#/home/upup/TransferLearning-DogClassification/dog_data/val/n02105855-Shetland_sheepdog/n02105855_11876.jpg

import rospy

import sys
import numpy as np

from sensor_msgs.msg import Image as SensorImage

from PIL import Image as PILImage

# def main(argv):
#     print(argv)
#     # print(argv[1])
#     # print(argv[2])
#     # print(argv[3])

def read_file(img_path):
    print("[1] Read image from file")
    print(img_path)

    im = PILImage.open(img_path) 
    im.show()    
    # # im.save('./save.png')
    # im_array = np.array(im) #np.array(); np.asarray()
    # # print(im.shape)
    # print(im_array.shape)

    im = im.convert('RGB')

    #Convert [PIL image] to [sensor_msgs Image]
    msg = SensorImage()
    msg.header.stamp = rospy.Time.now()
    msg.height = im.height
    msg.width = im.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * im.width
    msg.data = np.array(im).tobytes()
    cnt = 0
    state = 1
    while(state):
        pub_img.publish(msg)
        cnt+=1
        if(cnt>1000):
            state = 0

        print("pub DONE")
        
def stream_webcam(data):
    print("[2] Stream webcam")
    #https://www.ncnynl.com/archives/201703/1437.html

def realsense_cb(data):
    print("realsense_cb")
    pub_img.publish(data)

def stream_realsense():
    print("[3] Stream Realsense")    
    sub_markers = rospy.Subscriber('/camera/color/image_raw', SensorImage, realsense_cb)
 
if __name__ == "__main__":
    
    rospy.init_node("img_node")
    pub_img = rospy.Publisher('publish_img', SensorImage)#, queue_size = 10)

    print("Select image input method: [1]from file, [2]stream webcam, [3]stream Realsense")
    method_type = int(input())
        
    if(method_type == 1):
        print("Enter image path:")
        img_path = input()
        read_file(img_path)

    elif(method_type == 2):
        stream_webcam()

    elif(method_type == 3):
        stream_realsense()

    else:
        print("ERROR! Wrong input image method_type!")
    
    rospy.spin()
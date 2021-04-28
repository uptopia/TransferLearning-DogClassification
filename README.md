#Transfer Learning for Dog Classification

##Dependencies
ubuntu 18.04
python 3.6.9
pytorch 1.8.1+cpu
opencv 4.5.1
ROS melodic
cv_bridge (自己編譯)
realsense sdk
realsense_ros

##Execution
terminal 1
```
$roscore
```

terminal 2
```
$ cd <realsense_ros>
$. devel/setup.bash
$roslaunch realsense2_camera rs_camera.launch
```

terminal 3
```
$cd <TransferLearning-DogClassification>
$. devel/setup.bash
$rosrun publish_img publish_img.py
```
[1] img 
[2] webcam (TODO)
[3] realsense

terminal 4
```
$cd <TransferLearning-DogClassification>
$. devel/setup.bash
$rosrun classify_dogs dog_predict.py
```